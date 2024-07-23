from typing import Any, Dict

import requests
# from async_lru import alru_cache <-- not used and missing in python env fsr

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
