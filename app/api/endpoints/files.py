import os
from datetime import datetime, timezone

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import config, supabase
from app.schemas.clay import FileDeleted, FileObject
from app.utils.misc import random_string

router = APIRouter()


@router.post("")
async def upload_file(file: UploadFile = File(...)) -> FileObject:

    file_id = f"file-{random_string()}"
    temp_file_path = config.FILES_CACHE_DIR / file_id

    with open(temp_file_path, "wb") as temp_file:
        while content := await file.read(10 * 2**20):  # Read in chunks of 10 MB
            temp_file.write(content)

    with open(temp_file_path, "rb") as temp_file:
        supabase.storage.from_(config.SUPABASE_FILES_BUCKET).upload(path=file_id, file=temp_file)

    created_at = datetime.now(timezone.utc)
    metadata = {
        "id": file_id,
        "bytes": os.path.getsize(temp_file_path),
        "created_at": created_at.isoformat(),
        "filename": file.filename,
    }
    supabase.table(config.SUPABASE_FILES_METADATA_TABLE).insert(metadata).execute()

    metadata["created_at"] = int(created_at.timestamp())

    # os.remove(temp_file_path)

    return FileObject(**metadata)


@router.get("/{file_id}")
async def retrieve_file(file_id: str) -> FileObject:

    result = supabase.table(config.SUPABASE_FILES_METADATA_TABLE).select("*").eq("id", file_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="File not found")

    metadata = result.data[0]
    metadata["created_at"] = int(datetime.fromisoformat(metadata["created_at"]).timestamp())

    return FileObject(**metadata)


@router.delete("/{file_id}")
async def delete_file(file_id: str) -> FileDeleted:

    result = supabase.table("files_metadata").select("*").eq("id", file_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="File not found")

    cache_path = config.FILES_CACHE_DIR / file_id
    cache_path.unlink(missing_ok=True)

    try:
        supabase.storage.from_(config.SUPABASE_FILES_BUCKET).remove(file_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete file from storage")

    delete_result = supabase.table(config.SUPABASE_FILES_METADATA_TABLE).delete().eq("id", file_id).execute()
    if not delete_result.data:
        raise HTTPException(status_code=500, detail="Failed to delete file metadata")

    return FileDeleted(id=file_id, deleted=True)
