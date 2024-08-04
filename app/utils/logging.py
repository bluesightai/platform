import json
import time
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
