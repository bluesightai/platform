import os
from pathlib import Path
from typing import Literal, Tuple

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.api.deps import SessionDep
from app.config import config
from app.crud.model_metadata import crud_model_metadata
from app.schemas.model_metadata import ModelMetadata, ModelMetadataCreate, ModelMetadataUpdate
from app.utils.logging import LoggingRoute

router = APIRouter(route_class=LoggingRoute)


async def upload_model(
    session: SessionDep, task: Literal["classification", "segmentation"], local_model_path: Path
) -> ModelMetadata:
    model_metadata = await crud_model_metadata.create(db=session, obj_in=ModelMetadataCreate(task=task))
    try:
        with open(local_model_path, "rb") as f:
            await session.storage.from_(config.SUPABASE_MODELS_BUCKET).upload(path=model_metadata.id, file=f)

        model_metadata = await crud_model_metadata.update(
            db=session, id=model_metadata.id, obj_in=ModelMetadataUpdate(bytes=os.path.getsize(local_model_path))
        )

        local_model_path = local_model_path.rename(config.CACHE_DIR / model_metadata.id)
        # local_model_path.unlink(missing_ok=True)
    except Exception as e:
        await crud_model_metadata.delete(db=session, id=model_metadata.id)
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {e.__class__.__name__} {e}")
    return model_metadata


async def download_model(session: SessionDep, model_id: str) -> Tuple[ModelMetadata, Path]:
    local_model_path = config.CACHE_DIR / model_id
    if not local_model_path.exists():
        logger.debug(f"Downloading model '{model_id}' from bucket")
        with open(local_model_path, "wb") as f:
            f.write(await session.storage.from_(config.SUPABASE_MODELS_BUCKET).download(path=model_id))
    return await retrieve_model_metadata(session, model_id), local_model_path


@router.get("/{model_id}")
async def retrieve_model_metadata(session: SessionDep, model_id: str) -> ModelMetadata:
    model_metadata = await crud_model_metadata.get(db=session, id=model_id)
    if not model_metadata:
        raise HTTPException(status_code=404, detail="Model not found")
    return model_metadata


@router.delete("/{model_id}")
async def delete_model(session: SessionDep, model_id: str) -> ModelMetadata:
    model_metadata = await retrieve_model_metadata(session, model_id)

    local_model_path = config.CACHE_DIR / model_id
    local_model_path.unlink(missing_ok=True)

    try:
        await session.storage.from_(config.SUPABASE_MODELS_BUCKET).remove([model_id])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model from storage: {e.__class__.__name__} {e} ")

    model_metadata = await crud_model_metadata.delete(db=session, id=model_id)
    return model_metadata
