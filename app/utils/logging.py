import json
import time
from typing import Any, Callable, Dict

from fastapi import HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from fastapi.routing import APIRoute
from loguru import logger
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from app.config import supabase
from app.utils.print import truncating_pformat
from app.utils.requests import fetch_ip_data


# maybe async? probably no need as it runs in the background
def log_request(request_data: Dict[str, Any]):
    # user

    request_data["body"] = json.loads(request_data["body"].decode("utf-8") or "{}")
    request_data["response"] = json.loads(request_data["response"].decode("utf-8"))

    # logger.info(pformat(request_data, compact=True))
    logger.info(truncating_pformat(request_data))

    # await asyncio.sleep(3)
    # time.sleep(5)
    # request_data["ip"] = "82.163.218.35"

    if not supabase.table("ip_data").select("*").eq("ip", request_data["ip"]).execute().data:
        logger.info(f"ip {request_data['ip']} is not present, retrieving it...")
        # TODO: error handler
        ip_data = fetch_ip_data(request_data["ip"])
        supabase.table("ip_data").insert({"ip": request_data["ip"], "data": ip_data}).execute()

    supabase.table("requests").insert(request_data).execute()


class LoggingRoute(APIRoute):
    """
    https://stackoverflow.com/questions/69670125/how-to-log-raw-http-request-response-in-python-fastapi/73464007#73464007
    """

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:

            start_time = time.time()
            try:
                response = await original_route_handler(request)
            except HTTPException as e:
                logger.error("Got HTTPException!")
                response = PlainTextResponse(
                    content=json.dumps({"detail": e.detail}).encode(),
                    status_code=e.status_code,
                    headers=e.headers,
                )
            end_time = time.time()

            tasks = response.background

            request_data = {
                "ip": request.client.host,
                "headers": dict(request.headers),
                "method": request.method,
                "url": str(request.url),
                "query_params": dict(request.query_params),
                "body": await request.body(),
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
