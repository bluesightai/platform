import os

from pydantic_settings import BaseSettings
from supabase import AClient, Client, create_client


class Config(BaseSettings):

    SUPABASE_URL: str = os.environ.get("SUPABASE_URL") or ""
    SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY") or ""
    SUPABASE_MODEL_BUCKET: str = "models"


config = Config()

supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
