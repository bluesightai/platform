from supabase.client import AsyncClient

from app.config import config
from app.crud.base import CRUDBase
from app.schemas.request_metadata import RequestMetadata, RequestMetadataCreate


class CRUDRequestMetadata(CRUDBase[RequestMetadata, RequestMetadataCreate, RequestMetadataCreate]):
    async def create(self, db: AsyncClient, *, obj_in: RequestMetadataCreate) -> RequestMetadata:
        return await super().create(db, obj_in=obj_in)


crud_request_metadata = CRUDRequestMetadata(RequestMetadata, config.SUPABASE_REQUESTS_METADATA_TABLE)
