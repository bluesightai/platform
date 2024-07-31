from pathlib import Path
from typing import Any, Dict

import aiohttp
import requests
from async_lru import alru_cache

# @alru_cache
# async def fetch_ip_data(ip: str) -> Dict[str, Any]:
#     url = f"http://ip-api.com/json/{ip}"
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url) as response:
#             if response.status == 200:
#                 data = await response.json()
#             else:
#                 data = {"query": ip}
#                 print(f"Request failed with status code: {response.status}")
#     return data


# @lru_cache()
def fetch_ip_data(ip: str) -> Dict[str, Any]:
    url = f"http://ip-api.com/json/{ip}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
    else:
        data = {"query": ip}
        print(f"Request failed with status code: {response.status_code}")
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
