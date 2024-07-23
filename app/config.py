import os

from pydantic_settings import BaseSettings
from supabase import AClient, Client, create_client


class Config(BaseSettings):

    SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")
    SUPABASE_MODEL_BUCKET: str = "models"


config = Config()

try:
    supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
except: # initializing in global scope make it difficult to import stuff outside of the module
    supabase = None
