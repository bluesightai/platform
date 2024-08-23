import asyncio
import math
import os
import traceback
import urllib.parse
from io import BytesIO
from pathlib import Path
from pprint import pformat
from typing import List, Literal, Optional, Tuple, TypedDict, Union

import aiohttp
import asyncpg
import matplotlib.pyplot as plt
import numpy as np
from fire import Fire
from loguru import logger
from PIL import Image
from shapely import geometry, wkt
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm


class TileData(TypedDict):
    tile: Image.Image
    nw_lat: float
    nw_lng: float
    se_lat: float
    se_lng: float


class StyleDict(TypedDict, total=False):
    stylers: List[dict]
    featureType: str
    elementType: str


class SessionResponse(TypedDict):
    session: str
    expiry: str
    tileWidth: int
    tileHeight: int
    imageFormat: str


def bbox_to_wkt(bbox: tuple[float, float, float, float]) -> str:
    """Convert a bounding box to a WKT string"""
    minx, miny, maxx, maxy = bbox
    polygon = geometry.box(minx, miny, maxx, maxy)
    return polygon.wkt


def from_lat_lng_to_point(lat: float, lng: float, tile_size: int) -> Tuple[float, float]:
    """Convert latitude and longitude to point coordinates."""
    siny = math.sin((lat * math.pi) / 180)
    siny = min(max(siny, -0.9999), 0.9999)

    x = tile_size * (0.5 + lng / 360)
    y = tile_size * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))

    return (x, y)


def from_lat_lng_to_tile_coord(lat: float, lng: float, zoom: int, tile_size: int) -> Tuple[int, int]:
    """Convert latitude and longitude to tile coordinates."""
    x, y = from_lat_lng_to_point(lat, lng, tile_size)
    scale = 1 << zoom

    return int(x * scale / tile_size), int(y * scale / tile_size)


def from_tile_coord_to_lat_lng(x: int, y: int, zoom: int, tile_size: int) -> Tuple[float, float]:
    scale = 1 << zoom
    world_coord = (x * tile_size / scale, y * tile_size / scale)

    lng = world_coord[0] / tile_size * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * world_coord[1] / tile_size))))

    return (lat, lng)


async def get_session_token(
    api_key: str,
    map_type: Literal["roadmap", "satellite", "terrain", "streetview"] = "roadmap",
    language: str = "en-US",
    region: str = "US",
    image_format: Optional[Literal["jpeg", "png"]] = None,
    scale: Literal["scaleFactor1x", "scaleFactor2x", "scaleFactor4x"] = "scaleFactor1x",
    high_dpi: bool = False,
    layer_types: Optional[List[Literal["layerRoadmap", "layerStreetview", "layerTraffic"]]] = None,
    styles: Optional[List[StyleDict]] = None,
    overlay: Optional[bool] = None,
    api_options: Optional[List[str]] = None,
) -> Optional[SessionResponse]:
    """
    Retrieves a session token from the Google Maps Platform Map Tiles API.

    Args:
    api_key: Your Google Maps Platform API key
    map_type: The type of base map
    language: IETF language tag
    region: CLDR region identifier
    image_format: Specifies the file format to return
    scale: Scales-up the size of map elements
    high_dpi: Specifies whether to return high-resolution tiles
    layer_types: Array of layer types to add to the map
    styles: Array of JSON style objects for map customization
    overlay: Specifies whether layerTypes should be rendered as a separate overlay
    api_options: Array of additional API options

    Returns:
    A dictionary containing the session token and related information, or None if an error occurs
    """
    url = f"https://tile.googleapis.com/v1/createSession?key={api_key}"

    headers = {"Content-Type": "application/json"}

    payload = {"mapType": map_type, "language": language, "region": region, "scale": scale, "highDpi": high_dpi}

    if image_format:
        payload["imageFormat"] = image_format
    if layer_types:
        payload["layerTypes"] = layer_types
    if styles:
        payload["styles"] = styles
    if overlay is not None:
        payload["overlay"] = overlay
    if api_options:
        payload["apiOptions"] = api_options

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                return await response.json()
    except aiohttp.ClientError as e:
        logger.error(f"Error retrieving session token: {e}")
        return None


async def get_satellite_tile(
    session: aiohttp.ClientSession,
    session_token: str,
    api_key: str,
    zoom: int,
    x: int,
    y: int,
    cache_dir: Path = Path(".cache/gcp_tiles"),
) -> Optional[TileData]:
    """
    Retrieves a satellite tile from the Google Maps Platform Map Tiles API.

    Args:
    session: An aiohttp ClientSession object
    session_token: The session token obtained from get_session_token function
    api_key: Your Google Maps Platform API key
    zoom: The zoom level of the tile (typically 0-22, depends on the area)
    x: The x coordinate of the tile
    y: The y coordinate of the tile
    cache_dir_path: The path to the cache directory

    Returns:
    A dictionary containing the tile image and its bounding box coordinates, or None if an error occurs
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{zoom}_{x}_{y}.png"

    if cache_file.exists():
        logger.debug(f"Loading tile from cache: {cache_file}")
        img = Image.open(cache_file)
        nw_lat, nw_lng = from_tile_coord_to_lat_lng(x, y, zoom, img.width)
        se_lat, se_lng = from_tile_coord_to_lat_lng(x + 1, y + 1, zoom, img.width)
        return {"tile": img, "nw_lat": nw_lat, "nw_lng": nw_lng, "se_lat": se_lat, "se_lng": se_lng}

    url = f"https://tile.googleapis.com/v1/2dtiles/{zoom}/{x}/{y}"
    params = {"session": session_token, "key": api_key}

    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            content = await response.read()

        with open(cache_file, "wb") as f:
            f.write(content)
        logger.debug(f"Tile saved to cache: {cache_file}")

        img = Image.open(BytesIO(content))
        nw_lat, nw_lng = from_tile_coord_to_lat_lng(x, y, zoom, img.width)
        se_lat, se_lng = from_tile_coord_to_lat_lng(x + 1, y + 1, zoom, img.width)

        return {"tile": img, "nw_lat": nw_lat, "nw_lng": nw_lng, "se_lat": se_lat, "se_lng": se_lng}
    except aiohttp.ClientError as e:
        logger.error(f"Error retrieving satellite tile with x={x}, y={y}, zoom={zoom}: {e}")
        return None


def split_satellite_tile(tile_data: TileData, target_size: int) -> List[TileData]:
    """
    Splits a satellite tile into smaller tiles of the specified size.

    Args:
    tile_data: A dictionary containing the tile image and its bounding box coordinates
    target_size: The size of the smaller tiles

    Returns:
    A list of dictionaries containing the smaller tile images and their bounding box coordinates
    """
    img, nw_lat, nw_lng, se_lat, se_lng = (
        tile_data["tile"],
        tile_data["nw_lat"],
        tile_data["nw_lng"],
        tile_data["se_lat"],
        tile_data["se_lng"],
    )
    width, height = img.size

    if width != height:
        raise ValueError("Input image must be square")

    if width % target_size != 0:
        raise ValueError(f"Input image dimensions must be divisible by {target_size}")

    tiles_per_side = width // target_size
    lat_step = (se_lat - nw_lat) / tiles_per_side
    lng_step = (se_lng - nw_lng) / tiles_per_side

    result: List[TileData] = []

    for y in range(tiles_per_side):
        for x in range(tiles_per_side):
            left = x * target_size
            upper = y * target_size
            right = left + target_size
            lower = upper + target_size

            tile_img = img.crop((left, upper, right, lower))

            tile_nw_lat = nw_lat + y * lat_step
            tile_nw_lng = nw_lng + x * lng_step
            tile_se_lat = tile_nw_lat + lat_step
            tile_se_lng = tile_nw_lng + lng_step

            result.append(
                {
                    "tile": tile_img,
                    "nw_lat": tile_nw_lat,
                    "nw_lng": tile_nw_lng,
                    "se_lat": tile_se_lat,
                    "se_lng": tile_se_lng,
                }
            )

    return result


async def fetch_tiles(
    session_token: str,
    api_key: str,
    tile_size: int,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    zoom: int,
    cache_dir: Path = Path(".cache/gcp_tiles"),
) -> list[TileData]:
    """
    Fetches satellite tiles for a given bounding box.

    Args:
    session_token: The session token obtained from get_session_token function
    api_key: Your Google Maps Platform API keys
    tile_size: The size of the tiles
    start_lat: The latitude of the top-left corner of the bounding bbox
    start_lng: The longitude of the top-left corner of the bounding bbox
    end_lat: The latitude of the bottom-right corner of the bounding bbox
    end_lng: The longitude of the bottom-right corner of the bounding bbox
    zoom: The zoom level of the tiles

    Returns:
    A list of dictionaries containing the tile images and their bounding box coordinates
    """

    start_x, start_y = from_lat_lng_to_tile_coord(start_lat, start_lng, zoom, tile_size)
    end_x, end_y = from_lat_lng_to_tile_coord(end_lat, end_lng, zoom, tile_size)

    min_x, max_x = min(start_x, end_x), max(start_x, end_x)
    min_y, max_y = min(start_y, end_y), max(start_y, end_y)

    total_tiles = (max_x - min_x + 1) * (max_y - min_y + 1)
    logger.info(f"Fetching {total_tiles} tiles from GCP")

    async with aiohttp.ClientSession() as session:
        tasks = []
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                task = get_satellite_tile(
                    session=session,
                    session_token=session_token,
                    api_key=api_key,
                    zoom=zoom,
                    x=x,
                    y=y,
                    cache_dir=cache_dir,
                )
                tasks.append(task)

        tiles: List[TileData | None] = await async_tqdm.gather(*tasks, desc="Fetching tiles", total=len(tasks))

    filtered_tiles: List[TileData] = [tile for tile in tiles if tile is not None]
    logger.info(f"Retrieved {len(filtered_tiles)} out of {total_tiles} tiles after filtering out errors")

    return filtered_tiles


async def get_embeddings(session: aiohttp.ClientSession, images: List[Image.Image]) -> List[List[float]]:
    """Get embeddings for a batch of images using the bluesight.ai API asynchronously.

    Args:
    session: An aiohttp ClientSession object
    images: A list of PIL Image objects

    Returns:
    A list of embeddings for each image, or None if an error
    """
    url = "https://api.bluesight.ai/embeddings/img"

    payload = {
        "model": "clip",
        "images": [
            {
                "gsd": 0.6,
                "bands": ["red", "green", "blue"],
                "pixels": np.array(image.convert("RGB")).transpose(2, 0, 1).tolist(),
            }
            for image in images
        ],
    }
    headers = {"Content-Type": "application/json"}

    async with session.post(url, json=payload, headers=headers) as response:
        response.raise_for_status()
        return (await response.json())["embeddings"]


async def insert_to_postgres(
    conn: asyncpg.Connection,
    table_name: str,
    data: list[tuple[str, list[float]]],
) -> None:
    query = f"""
    INSERT INTO {table_name} (location, embedding)
    VALUES (ST_GeomFromText($1, 4326)::geography, $2)
    ON CONFLICT (location) DO NOTHING
    """

    # Convert data to the format expected by PostgreSQL
    prepared_data = [(bbox, f"[{','.join(map(str, embedding))}]") for bbox, embedding in data]

    await conn.executemany(query, prepared_data)


async def insert_area(
    start_lat: float = 37.811219311975265,
    start_lng: float = -122.52665573314543,
    end_lat: float = 37.69850383939589,
    end_lng: float = -122.34837526369923,
    zoom: int = 18,
    scale: Literal["scaleFactor1x", "scaleFactor2x", "scaleFactor4x"] = "scaleFactor4x",
    target_tile_size: int = 256,
    batch_size: int = 256,
    table_name: str = "clip_boxes_gcp",
):
    """Insert satellite tiles and embeddings for a specified area into a PostgreSQL database.

    Args:
    start_lat: The latitude of the top-left corner of the bounding bbox
    start_lng: The longitude of the top-left corner of the bounding bbox
    end_lat: The latitude of the bottom-right corner of the bounding bbox
    end_lng: The longitude of the bottom-right corner of the bounding bbox
    zoom: The zoom level of the tiles
    scale: Scales-up the size of map elements
    target_tile_size: The size of the smaller tiles
    batch_size: The number of tiles to process and insert into the database at a time
    table_name: The name of the table to insert the data into
    """

    GCP_API_KEY = os.getenv("GCP_API_KEY")
    if not GCP_API_KEY:
        raise ValueError("Please set the GCP_API_KEY environment variable")

    SUPABASE_POSTGRES_PASSWORD = os.getenv("SUPABASE_POSTGRES_PASSWORD")
    if not SUPABASE_POSTGRES_PASSWORD:
        raise ValueError("Please set the SUPABASE_POSTGRES_PASSWORD environment variable")
    postgres_uri = f"postgresql://postgres.biccczfztgnfaqzmizan:{urllib.parse.quote_plus(SUPABASE_POSTGRES_PASSWORD)}@aws-0-us-east-1.pooler.supabase.com:6543/postgres"

    cache_dir = Path(f".cache/gcp_tiles/{scale}")

    session_data = await get_session_token(
        api_key=GCP_API_KEY,
        map_type="satellite",
        language="en-US",
        region="US",
        image_format="png",
        scale=scale,
        high_dpi=True,
    )
    if not session_data:
        raise ValueError("Error retrieving session data")
    logger.info(f"GCP session data:\n{pformat(session_data)}")
    session_token = session_data.get("session")
    tile_size = session_data.get("tileWidth")
    logger.info(f"Retrieving tiles with size {tile_size}, scale {scale}, zoom {zoom}, cache dir {cache_dir}")

    tiles = await fetch_tiles(
        session_token=session_token,
        api_key=GCP_API_KEY,
        tile_size=tile_size,
        start_lat=start_lat,
        start_lng=start_lng,
        end_lat=end_lat,
        end_lng=end_lng,
        zoom=zoom,
        cache_dir=cache_dir,
    )

    tiles = [
        tile
        for split_tile in tiles
        for tile in split_satellite_tile(tile_data=split_tile, target_size=target_tile_size)
    ]
    logger.info(f"Amount of tiles after splitting from {tile_size} to {target_tile_size}: {len(tiles)}")

    logger.info(f"Processing tiles and inserting into database in batches of {batch_size}")
    conn = await asyncpg.connect(postgres_uri, statement_cache_size=0)
    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(tiles), batch_size), desc="Processing and inserting batches", unit="batch"):
            batch_tiles = tiles[i : i + batch_size]

            # Fetch embeddings for the current batch of tiles
            batch_images = [tile["tile"] for tile in batch_tiles]
            batch_embeddings = await get_embeddings(session=session, images=batch_images)

            # Prepare data for database insertion
            data = [
                (bbox_to_wkt((tile["nw_lng"], tile["nw_lat"], tile["se_lng"], tile["se_lat"])), embedding)
                for tile, embedding in zip(batch_tiles, batch_embeddings)
            ]

            # Insert the batch into the database
            await insert_to_postgres(conn=conn, table_name=table_name, data=data)

    await conn.close()

    # # Calculate the dimensions of the final image
    # width = sum(tile.width for (tile, _, _) in tiles[0])
    # height = sum(row[0][0].height for row in tiles)
    # # Create a new image to hold all tiles
    # final_image = Image.new("RGB", (width, height))
    # # Paste all tiles into the final image
    # y_offset = 0
    # for row in tiles:
    #     x_offset = 0
    #     for tile, _, _ in row:
    #         final_image.paste(tile, (x_offset, y_offset))
    #         x_offset += tile.width
    #     y_offset += row[0][0].height
    # return final_image

    # lat, lng = 33.75680859184824, -118.23377899768916
    # x, y = from_lat_lng_to_tile_coord(lat, lng, zoom, tile_size)
    # async with aiohttp.ClientSession() as session:
    #     tile_data = await get_satellite_tile(
    #         session=session,
    #         session_token=session_token,
    #         api_key=GCP_API_KEY,
    #         zoom=zoom,
    #         x=x,
    #         y=y,
    #         cache_dir=cache_dir,
    #     )
    # if not tile_data:
    #     raise ValueError("Error retrieving tile data")
    # img, nw_lat, nw_lng, se_lat, se_lng = (
    #     tile_data["tile"],
    #     tile_data["nw_lat"],
    #     tile_data["nw_lng"],
    #     tile_data["se_lat"],
    #     tile_data["se_lng"],
    # )
    # logger.debug(f"Bbox: {nw_lat:.6f}, {nw_lng:.6f}, {se_lat:.6f}, {se_lng:.6f}")
    # tiles = split_satellite_tile(tile_data=tile_data, target_size=256)
    # logger.info(f"Split into {len(tiles)} tiles")
    # plt.imshow(img)
    # plt.axis("off")
    # plt.show()


if __name__ == "__main__":
    Fire(insert_area)
