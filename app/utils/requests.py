from pathlib import Path
from typing import Any, Dict

import aiohttp
import requests
from loguru import logger
from supabase.client import AsyncClient

from app.config import config


async def fetch_ip_data(ip: str) -> Dict[str, Any]:
    url = f"http://ip-api.com/json/{ip}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
            else:
                data = {"query": ip}
                print(f"IP request failed with status code: {response.status} and response: {response.text}")
    return data


async def download_file_in_chunks(url: str, path: Path, chunk_size: int = 10 * 2**20):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as res:
            res.raise_for_status()
            with open(path, "wb+") as f:
                while True:
                    chunk = await res.content.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)


async def download_file_from_bucket(session: AsyncClient, file_id: str) -> Path:

    local_file_path = config.CACHE_DIR / file_id
    if not local_file_path.exists():
        logger.info(f"Downloading file {file_id} from bucket")
        download_url = await session.storage.from_(config.SUPABASE_FILES_BUCKET).get_public_url(file_id)
        await download_file_in_chunks(url=download_url, path=local_file_path)
    return local_file_path
