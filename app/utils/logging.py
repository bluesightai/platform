import json
import time
from typing import Any, Callable, Dict

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import PlainTextResponse
from fastapi.routing import APIRoute
from loguru import logger
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from app.config import config, supabase
from app.utils.print import truncating_pformat
from app.utils.requests import fetch_ip_data


def delete_keys(d, keys=("pixels", "embeddings", "labels")):
    if isinstance(d, dict):
        keys_to_delete = [key for key in d if key in keys]
        for key in keys_to_delete:
            del d[key]
        for value in d.values():
            delete_keys(value)
    elif isinstance(d, list):
        for item in d:
            delete_keys(item)


# maybe async? probably no need as it runs in the background
def log_request(request_data: Dict[str, Any]):
    # user

    request_data["body"] = json.loads(request_data["body"].decode("utf-8") if request_data["body"] else "{}" or "{}")
    request_data["response"] = json.loads(request_data["response"].decode("utf-8"))

    delete_keys(request_data)

    # logger.info(truncating_pformat(request_data))

    if not supabase.table(config.SUPABASE_IP_DATA_TABLE).select("*").eq("ip", request_data["ip"]).execute().data:
        logger.info(f"ip {request_data['ip']} is not present, retrieving it...")
        # TODO: error handler
        ip_data = fetch_ip_data(request_data["ip"])
        supabase.table(config.SUPABASE_IP_DATA_TABLE).insert({"ip": request_data["ip"], "data": ip_data}).execute()

    supabase.table(config.SUPABASE_REQUESTS_TABLE).insert(request_data).execute()


class LoggingRoute(APIRoute):
    """
    https://stackoverflow.com/questions/69670125/how-to-log-raw-http-request-response-in-python-fastapi/73464007#73464007
    """

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:

            start_time = time.time()
            logger.info(f"Received {request.method} request for {request.url}")
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

            request_data = {
                "ip": request.client.host,
                "headers": dict(request.headers),
                "method": request.method,
                "url": str(request.url),
                "query_params": dict(request.query_params),
                "body": None,
                "response_status_code": response.status_code,
                "response": response.body,
                "process_time": round(end_time - start_time, 3),
            }

            if isinstance(response, StreamingResponse):
                res_body = b""
                async for item in response.body_iterator:
                    res_body += item

                request_data["response"] = res_body

                task = BackgroundTask(log_request, request_data)
                response = Response(
                    content=res_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
            else:
                task = BackgroundTask(log_request, request_data)

            # check if the original response had background tasks already attached to it
            if tasks:
                tasks.add_task(task)  # add the new task to the tasks list
                response.background = tasks
            else:
                response.background = task

            return response

        return custom_route_handler
