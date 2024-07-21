import os

from pydantic_settings import BaseSettings
from supabase import AClient, Client, create_client
from supabase.client import ClientOptions

class Config(BaseSettings):

    SUPABASE_URL: str = os.environ.get("SUPABASE_URL") or ""
    SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY") or ""
    SUPABASE_MODEL_BUCKET: str = "models"


config = Config()

supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY, options=ClientOptions(
    postgrest_client_timeout=999999999,
    storage_client_timeout=999999999
  ))
