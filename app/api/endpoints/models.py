import os

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.deps import SessionDep
from app.config import config
from app.crud.model_metadata import crud_model_metadata
from app.schemas.model_metadata import ModelMetadata, ModelMetadataCreate, ModelMetadataUpdate
from app.utils.logging import LoggingRoute

router = APIRouter(route_class=LoggingRoute)


@router.post("", include_in_schema=False)
async def upload_model(
    session: SessionDep, metadata: ModelMetadataCreate, file: UploadFile = File(...)
) -> ModelMetadata:
    model_metadata = await crud_model_metadata.create(db=session, obj_in=metadata)
    try:
        local_model_path = config.CACHE_DIR / model_metadata.id
        with open(local_model_path, "wb") as f:
            while content := await file.read(10 * 2**20):  # Read in chunks of 10 MB
                f.write(content)

        with open(local_model_path, "rb") as f:
            await session.storage.from_(config.SUPABASE_MODELS_BUCKET).upload(path=model_metadata.id, file=f)

        model_metadata = await crud_model_metadata.update(
            db=session, id=model_metadata.id, obj_in=ModelMetadataUpdate(bytes=os.path.getsize(local_model_path))
        )
        # local_model_path.unlink(missing_ok=True)
    except Exception as e:
        await crud_model_metadata.delete(db=session, id=model_metadata.id)
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {e.__class__.__name__} {e}")
    return model_metadata


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
