from typing import Annotated

from fastapi import Depends
from supabase.client import AsyncClient, ClientOptions

from app.config import config
from supabase import acreate_client

super_client: AsyncClient | None = None


async def init_super_client() -> None:
    global super_client
    super_client = await acreate_client(
        config.SUPABASE_URL,
        config.SUPABASE_KEY,
        options=ClientOptions(postgrest_client_timeout=10**9, storage_client_timeout=10**9),
    )


def get_super_client() -> AsyncClient:
    if super_client is None:
        raise RuntimeError("Supabase client is not initialized")
    return super_client


async def get_db() -> AsyncClient:
    """
    Later this will be dependent on the Token and we
    will return a client based on the user's token
    with the appropriate permissions
    """

    if super_client is None:
        raise RuntimeError("Supabase client is not initialized")
    return super_client


SessionDep = Annotated[AsyncClient, Depends(get_db)]
