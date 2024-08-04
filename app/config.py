import os
from pathlib import Path

from pydantic_settings import BaseSettings
from supabase.client import ClientOptions

from supabase import Client, create_client


class Config(BaseSettings):

    SUPABASE_URL: str = os.environ.get("SUPABASE_URL") or ""
    SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY") or ""
    SUPABASE_REQUESTS_TABLE: str = "requests"
    SUPABASE_IP_DATA_TABLE: str = "ip_data"
    SUPABASE_FILES_METADATA_TABLE: str = "files_metadata"
    SUPABASE_MODELS_METADATA_TABLE: str = "models_metadata"
    SUPABASE_TRAINING_JOBS_TABLE: str = "training_jobs"
    SUPABASE_MODELS_BUCKET: str = "models"
    SUPABASE_FILES_BUCKET: str = "files"
    CACHE_DIR: Path = Path(".cache/")
    CHECKPOINTS_DIR: Path = Path("checkpoints/")


config = Config()
config.CACHE_DIR.mkdir(exist_ok=True)

supabase: Client = create_client(
    config.SUPABASE_URL,
    config.SUPABASE_KEY,
    options=ClientOptions(postgrest_client_timeout=10**9, storage_client_timeout=10**9),
)
