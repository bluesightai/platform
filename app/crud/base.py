from datetime import datetime
from typing import Any, Dict, Generic, TypeVar

from pydantic import BaseModel
from supabase.client import AsyncClient

ModelType = TypeVar("ModelType", bound=BaseModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, model: type[ModelType], table_name: str):
        self.model = model
        self.table_name = table_name

    @staticmethod
    def convert_datetime_to_int(data: Dict[str, Any]) -> Dict[str, Any]:
        for key in ["created_at", "finished_at"]:
            if key in data and data[key]:
                data[key] = int(datetime.strptime(data[key], "%Y-%m-%dT%H:%M:%S.%f%z").timestamp())
        return data

    @staticmethod
    def convert_float_to_datetime(obj: UpdateSchemaType) -> UpdateSchemaType:
        for key in ["created_at", "finished_at"]:
            if hasattr(obj, key) and getattr(obj, key):
                setattr(obj, key, datetime.fromtimestamp(getattr(obj, key)).isoformat())
        return obj

    async def create(self, db: AsyncClient, *, obj_in: CreateSchemaType) -> ModelType:
        """create by CreateSchemaType"""
        data, count = await db.table(self.table_name).insert(obj_in.model_dump()).execute()
        _, created = data
        created[0] = self.convert_datetime_to_int(created[0])
        return self.model(**created[0])

    async def get(self, db: AsyncClient, *, id: str) -> ModelType | None:
        """get by table_name by id"""
        data, count = await db.table(self.table_name).select("*").eq("id", id).execute()
        _, got = data
        if not got:
            return None
        got[0] = self.convert_datetime_to_int(got[0])
        return self.model(**got[0])

    async def update(self, db: AsyncClient, *, id: str, obj_in: UpdateSchemaType) -> ModelType:
        """update by UpdateSchemaType"""
        obj_in = self.convert_float_to_datetime(obj_in)
        data, count = await db.table(self.table_name).update(obj_in.model_dump()).eq("id", id).execute()
        _, updated = data
        updated[0] = self.convert_datetime_to_int(updated[0])
        return self.model(**updated[0])

    async def delete(self, db: AsyncClient, *, id: str) -> ModelType:
        """remove by UpdateSchemaType"""
        data, count = await db.table(self.table_name).delete().eq("id", id).execute()
        _, deleted = data
        deleted[0] = self.convert_datetime_to_int(deleted[0])
        return self.model(**deleted[0])

    # async def get_all(self, db: AsyncClient) -> list[ModelType]:
    #     """get all by table_name"""
    #     data, count = await db.table(self.table_name).select("*").execute()
    #     _, got = data
    #     return [self.model(**item) for item in got]

    # async def get_multi_by_owner(self, db: AsyncClient, *, user: UserIn) -> list[ModelType]:
    #     """get by owner,use it  if rls failed to use"""
    #     data, count = await db.table(self.table_name).select("*").eq("user_id", user.id).execute()
    #     _, got = data
    #     return [self.model(**item) for item in got]
