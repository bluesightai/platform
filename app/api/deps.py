from typing import Annotated

from fastapi import Depends
from supabase.client import AsyncClient, ClientOptions

from app.config import config
from supabase import acreate_client

super_client: AsyncClient


async def init_super_client() -> None:
    global super_client
    super_client = await acreate_client(
        config.SUPABASE_URL,
        config.SUPABASE_KEY,
        options=ClientOptions(postgrest_client_timeout=10, storage_client_timeout=10),
    )


async def get_db() -> AsyncClient:
    return super_client


SessionDep = Annotated[AsyncClient, Depends(get_db)]
