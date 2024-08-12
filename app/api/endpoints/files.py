import os
from typing import Annotated

import fastapi
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.deps import SessionDep
from app.config import config
from app.crud.file_metadata import crud_file_metadata
from app.schemas.file_metadata import FileMetadata, FileMetadataCreate, FileMetadataUpdate
from app.utils.logging import LoggingRoute

router = APIRouter(route_class=LoggingRoute)


@router.post("")
async def upload_file(
    session: SessionDep, file: UploadFile = File(..., description="The File object (not file name) to be uploaded. ")
) -> FileMetadata:
    """
    Upload a file to the Bluesight storage bucket.

    The file may be used for [creating a training job](https://docs.bluesight.ai/api-reference/training-jobs/create-training-job) or [inference](https://docs.bluesight.ai/api-reference/inference/run-trained-model-inference) (TBD).

    File is expected to be in HDF5 format. Check guides page for info on how to create a valid file and upload it (TBD).
    """
    file_metadata = await crud_file_metadata.create(db=session, obj_in=FileMetadataCreate(filename=file.filename))
    try:
        local_file_path = config.CACHE_DIR / file_metadata.id
        with open(local_file_path, "wb") as f:
            while content := await file.read(10 * 2**20):  # Read in chunks of 10 MB
                f.write(content)

        with open(local_file_path, "rb") as f:
            await session.storage.from_(config.SUPABASE_FILES_BUCKET).upload(path=file_metadata.id, file=f)

        file_metadata = await crud_file_metadata.update(
            db=session, id=file_metadata.id, obj_in=FileMetadataUpdate(bytes=os.path.getsize(local_file_path))
        )
        # local_file_path.unlink(missing_ok=True)
    except Exception as e:
        await crud_file_metadata.delete(db=session, id=file_metadata.id)
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {e.__class__.__name__} {e}")
    return file_metadata


@router.get("/{file_id}")
async def retrieve_file_metadata(
    session: SessionDep,
    file_id: Annotated[str, fastapi.Path(example="file-lw3zjxrg", description="The ID of the file")],
) -> FileMetadata:
    """
    Get info about a specific file.
    """
    file_metadata = await crud_file_metadata.get(db=session, id=file_id)
    if not file_metadata:
        raise HTTPException(status_code=404, detail="File not found")
    return file_metadata


@router.delete("/{file_id}")
async def delete_file(
    session: SessionDep,
    file_id: Annotated[str, fastapi.Path(example="file-lw3zjxrg", description="The ID of the file")],
) -> FileMetadata:
    """
    Delete a file.
    """
    file_metadata = await retrieve_file_metadata(session, file_id)

    local_file_path = config.CACHE_DIR / file_id
    local_file_path.unlink(missing_ok=True)

    try:
        await session.storage.from_(config.SUPABASE_FILES_BUCKET).remove([file_id])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file from storage: {e.__class__.__name__} {e} ")

    file_metadata = await crud_file_metadata.delete(db=session, id=file_id)
    return file_metadata
