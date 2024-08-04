from supabase.client import AsyncClient

from app.config import config
from app.crud.base import CRUDBase
from app.schemas.file_metadata import FileMetadata, FileMetadataCreate, FileMetadataUpdate


class CRUDFileMetadata(CRUDBase[FileMetadata, FileMetadataCreate, FileMetadataUpdate]):
    async def create(self, db: AsyncClient, *, obj_in: FileMetadataCreate) -> FileMetadata:
        return await super().create(db, obj_in=obj_in)

    async def get(self, db: AsyncClient, *, id: str) -> FileMetadata | None:
        return await super().get(db, id=id)

    async def update(self, db: AsyncClient, *, id: str, obj_in: FileMetadataUpdate) -> FileMetadata:
        return await super().update(db, id=id, obj_in=obj_in)

    async def delete(self, db: AsyncClient, *, id: str) -> FileMetadata:
        return await super().delete(db, id=id)

    # async def get_all(self, db: AsyncClient) -> list[FileMetadata]:
    #     return await super().get_all(db)

    # async def get_multi_by_owner(self, db: AsyncClient, *, user: UserIn) -> list[FileMetadata]:
    #     return await super().get_multi_by_owner(db, user=user)


crud_file_metadata = CRUDFileMetadata(FileMetadata, config.SUPABASE_FILES_METADATA_TABLE)
