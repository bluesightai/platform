from typing import Any

from loguru import logger

from app.api.deps import SessionDep
from app.crud.ip_metadata import crud_ip_metadata
from app.crud.request_metadata import crud_request_metadata
from app.schemas.ip_metadata import IPMetadataCreate
from app.schemas.request_metadata import RequestMetadata, RequestMetadataCreate
from app.utils.requests import fetch_ip_data


def delete_keys(d: dict[str, Any] | list[dict[str, Any]] | None, keys=("pixels", "embeddings", "labels")):
    if isinstance(d, dict):
        keys_to_delete = [key for key in d if key in keys]
        for key in keys_to_delete:
            del d[key]
        for value in d.values():
            delete_keys(value)
    elif isinstance(d, list):
        for item in d:
            delete_keys(item)


async def upload_request(session: SessionDep, request_metadata: RequestMetadataCreate) -> RequestMetadata:
    delete_keys(request_metadata.body)
    delete_keys(request_metadata.response)

    if not (await crud_ip_metadata.get(db=session, id=request_metadata.ip)):
        logger.debug(f"ip '{request_metadata.ip}' is not present, retrieving it...")
        ip_data = await fetch_ip_data(request_metadata.ip)
        await crud_ip_metadata.create(db=session, obj_in=IPMetadataCreate(id=request_metadata.ip, data=ip_data))

    request_metadata = await crud_request_metadata.create(db=session, obj_in=request_metadata)
    return request_metadata
