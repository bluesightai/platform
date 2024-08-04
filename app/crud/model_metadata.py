from supabase.client import AsyncClient

from app.config import config
from app.crud.base import CRUDBase
from app.schemas.model_metadata import ModelMetadata, ModelMetadataCreate, ModelMetadataUpdate


class CRUDModelMetadata(CRUDBase[ModelMetadata, ModelMetadataCreate, ModelMetadataUpdate]):
    async def create(self, db: AsyncClient, *, obj_in: ModelMetadataCreate) -> ModelMetadata:
        return await super().create(db, obj_in=obj_in)

    async def get(self, db: AsyncClient, *, id: str) -> ModelMetadata | None:
        return await super().get(db, id=id)

    async def update(self, db: AsyncClient, *, id: str, obj_in: ModelMetadataUpdate) -> ModelMetadata:
        return await super().update(db, id=id, obj_in=obj_in)

    async def delete(self, db: AsyncClient, *, id: str) -> ModelMetadata:
        return await super().delete(db, id=id)

    # async def get_all(self, db: AsyncClient) -> list[ModelMetadata]:
    #     return await super().get_all(db)

    # async def get_multi_by_owner(self, db: AsyncClient, *, user: UserIn) -> list[ModelMetadata]:
    #     return await super().get_multi_by_owner(db, user=user)


crud_model_metadata = CRUDModelMetadata(ModelMetadata, config.SUPABASE_MODELS_METADATA_TABLE)
