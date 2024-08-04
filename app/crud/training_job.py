from supabase.client import AsyncClient

from app.config import config
from app.crud.base import CRUDBase
from app.schemas.training_job import TrainingJob, TrainingJobCreate, TrainingJobUpdate


class CRUDTrainingJob(CRUDBase[TrainingJob, TrainingJobCreate, TrainingJobUpdate]):
    async def create(self, db: AsyncClient, *, obj_in: TrainingJobCreate) -> TrainingJob:
        return await super().create(db, obj_in=obj_in)

    async def get(self, db: AsyncClient, *, id: str) -> TrainingJob | None:
        return await super().get(db, id=id)

    async def update(self, db: AsyncClient, *, id: str, obj_in: TrainingJobUpdate) -> TrainingJob:
        return await super().update(db, id=id, obj_in=obj_in)

    # async def get_all(self, db: AsyncClient) -> list[TrainingJob]:
    #     return await super().get_all(db)

    # async def get_multi_by_owner(self, db: AsyncClient, *, user: UserIn) -> list[TrainingJob]:
    #     return await super().get_multi_by_owner(db, user=user)


crud_training_job = CRUDTrainingJob(TrainingJob, config.SUPABASE_TRAINING_JOBS_TABLE)
