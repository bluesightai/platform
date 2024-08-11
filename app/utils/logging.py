import http
import json
import logging
import os
import sys
import time
import warnings
from typing import Callable

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import PlainTextResponse
from fastapi.routing import APIRoute
from loguru import logger
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from app.api.deps import get_super_client
from app.api.endpoints.requests import upload_request
from app.schemas.request_metadata import RequestMetadataCreate


class LoggingRoute(APIRoute):
    """
    https://stackoverflow.com/questions/69670125/how-to-log-raw-http-request-response-in-python-fastapi/73464007#73464007
    """

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:

            start_time = time.time()
            logger.info(f"Received {request.method} request for {request.url}")

            super_client = get_super_client()
            try:
                response = await original_route_handler(request)
            except HTTPException as e:
                logger.error("Got HTTPException!")
                response = PlainTextResponse(
                    content=json.dumps({"detail": e.detail}).encode(),
                    status_code=e.status_code,
                    headers=e.headers,
                )
            except Exception as e:
                logger.exception(e)
                response = PlainTextResponse(
                    content=json.dumps({"detail": f"{e.__class__.__name__}: {str(e)}"}).encode(),
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
            end_time = time.time()

            tasks = response.background

            request_metadata = RequestMetadataCreate(
                ip=request.client.host if request.client else "",
                headers=dict(request.headers),
                method=request.method,
                url=str(request.url),
                query_params=dict(request.query_params),
                # body=json.loads(request.body.decode("utf-8")) if request.body else None,
                response_status_code=response.status_code,
                response=json.loads(response.body.decode("utf-8")),
                process_time=round(end_time - start_time, 3),
            )

            if isinstance(response, StreamingResponse):
                res_body = b""
                async for item in response.body_iterator:
                    res_body += item

                request_metadata.response = json.loads(res_body.decode("utf-8"))

                task = BackgroundTask(upload_request, super_client, request_metadata)
                response = Response(
                    content=res_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
            else:
                task = BackgroundTask(upload_request, super_client, request_metadata)

            # check if the original response had background tasks already attached to it
            if tasks:
                tasks.add_task(task)  # add the new task to the tasks list
                response.background = tasks
            else:
                response.background = task

            return response

        return custom_route_handler


def get_status_color(status_code: int) -> str:
    if status_code >= 500:
        return "\x1b[31m"  # Red
    elif status_code >= 400:
        return "\x1b[33m"  # Yellow
    elif status_code >= 300:
        return "\x1b[36m"  # Cyan
    elif status_code >= 200:
        return "\x1b[32m"  # Green
    else:
        return "\x1b[37m"  # White


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # Check if this is an access log
        if record.name == "uvicorn.access":
            try:
                # Parse the message to extract status code
                parts = record.getMessage().split()
                status_code = int(parts[-1])
                color = get_status_color(status_code)
                status_text = http.HTTPStatus(status_code).phrase

                # Format the message similar to uvicorn's format
                message = f'{" ".join(parts[:-1])} {color}{status_code} {status_text}\x1b[0m'

                logger.opt(depth=depth, exception=record.exc_info).log(level, message, process=os.getpid())
            except Exception:
                # If parsing fails, log the original message
                logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage(), process=os.getpid())
        else:
            # For non-access logs, include more detailed information
            logger.opt(depth=depth, exception=record.exc_info).log(
                level,
                record.getMessage(),
                file=record.filename,
                function=record.funcName,
                line=record.lineno,
                process=os.getpid(),
            )


def warn_with_loguru(message, category, filename, lineno, file=None, line=None):
    if file is not None:
        try:
            with open(filename) as f:
                source_line = f.readlines()[lineno - 1].strip()
        except (IOError, IndexError):
            source_line = None
    else:
        source_line = None

    warning_message = f"{filename}:{lineno}: {category.__name__}: {message}"
    if source_line:
        warning_message += f"\n  {source_line}"

    logger.opt(depth=2).warning(
        warning_message, file=filename, line=lineno, category=category.__name__, process=os.getpid()
    )


warnings.showwarning = warn_with_loguru

intercept_handler = InterceptHandler()
logging.getLogger("uvicorn").handlers = [InterceptHandler()]
logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
