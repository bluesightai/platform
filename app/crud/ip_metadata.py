from supabase.client import AsyncClient

from app.config import config
from app.crud.base import CRUDBase
from app.schemas.ip_metadata import IPMetadata, IPMetadataCreate


class CRUDIPMetadata(CRUDBase[IPMetadata, IPMetadataCreate, IPMetadataCreate]):
    async def create(self, db: AsyncClient, *, obj_in: IPMetadataCreate) -> IPMetadata:
        return await super().create(db, obj_in=obj_in)

    async def get(self, db: AsyncClient, *, id: str) -> IPMetadata | None:
        return await super().get(db, id=id)


crud_ip_metadata = CRUDIPMetadata(IPMetadata, config.SUPABASE_IP_DATA_TABLE)
