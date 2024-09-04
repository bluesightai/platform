import asyncio
import base64
import hashlib
import io
import math
import os
import urllib.parse
from io import BytesIO
from pathlib import Path
from pprint import pformat
from typing import AsyncGenerator, Literal, Optional, TypedDict

import aiohttp
import asyncpg
import numpy as np
from fire import Fire
from loguru import logger
from PIL import Image
from shapely import geometry
from supabase import acreate_client
from supabase.client import AsyncClient
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# Add this to the existing imports


class TileData(TypedDict):
    tile: Image.Image
    nw_lat: float
    nw_lng: float
    se_lat: float
    se_lng: float


class StyleDict(TypedDict, total=False):
    stylers: list[dict]
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


def mask_to_wkt(mask: NDArray[np.bool_], tile_data: TileData, bbox: tuple[int, int, int, int]) -> str:
    """Convert a segmentation mask to WKT format using tile data."""
    tile_width, tile_height = tile_data["tile"].size
    nw_lat, nw_lng, se_lat, se_lng = tile_data["nw_lat"], tile_data["nw_lng"], tile_data["se_lat"], tile_data["se_lng"]

    # Calculate lat/lng per pixel
    lat_per_pixel = (se_lat - nw_lat) / tile_height
    lng_per_pixel = (se_lng - nw_lng) / tile_width

    # Find contours in the mask
    contours = measure.find_contours(mask, 0.5)

    polygons: list[Polygon] = []
    for contour in contours:
        # Convert pixel coordinates to lat/lng
        coords = []
        for y, x in contour:
            lng = nw_lng + x * lng_per_pixel
            lat = nw_lat + y * lat_per_pixel
            coords.append((lng, lat))

        # Create a polygon from the coordinates
        poly = Polygon(coords)
        if poly.is_valid:
            polygons.append(poly)

    # # Create a MultiPolygon if there are multiple polygons
    # if len(polygons) > 1:
    #     multi_poly = MultiPolygon(polygons)
    # elif len(polygons) == 1:
    #     multi_poly = polygons[0]
    # else:
    #     raise ValueError("No valid polygons found in mask")

    if polygons:
        polygon = polygons[0]
    else:
        # Create a polygon that covers the entire tile if no valid polygons are found
        minx, miny, maxx, maxy = bbox
        polygon = geometry.box(minx, miny, maxx, maxy)

    return polygon.wkt


def from_lat_lng_to_point(lat: float, lng: float, tile_size: int) -> tuple[float, float]:
    """Convert latitude and longitude to point coordinates."""
    siny = math.sin((lat * math.pi) / 180)
    siny = min(max(siny, -0.9999), 0.9999)

    x = tile_size * (0.5 + lng / 360)
    y = tile_size * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))

    return (x, y)


def from_lat_lng_to_tile_coord(lat: float, lng: float, zoom: int, tile_size: int) -> tuple[int, int]:
    """Convert latitude and longitude to tile coordinates."""
    x, y = from_lat_lng_to_point(lat, lng, tile_size)
    scale = 1 << zoom

    return int(x * scale / tile_size), int(y * scale / tile_size)


def from_tile_coord_to_lat_lng(x: int, y: int, zoom: int, tile_size: int) -> tuple[float, float]:
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
    layer_types: Optional[list[Literal["layerRoadmap", "layerStreetview", "layerTraffic"]]] = None,
    styles: Optional[list[StyleDict]] = None,
    overlay: Optional[bool] = None,
    api_options: Optional[list[str]] = None,
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
        img = Image.open(cache_file).convert("RGB")
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

        img = Image.open(BytesIO(content)).convert("RGB")

        nw_lat, nw_lng = from_tile_coord_to_lat_lng(x, y, zoom, img.width)
        se_lat, se_lng = from_tile_coord_to_lat_lng(x + 1, y + 1, zoom, img.width)

        return {"tile": img, "nw_lat": nw_lat, "nw_lng": nw_lng, "se_lat": se_lat, "se_lng": se_lng}
    except aiohttp.ClientError as e:
        logger.error(f"Error retrieving satellite tile with x={x}, y={y}, zoom={zoom}: {e}")
        return None


def split_satellite_tile(tile_data: TileData, target_size: int) -> list[TileData]:
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

    result: list[TileData] = []

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
    target_tile_size: int,
    cache_dir: Path = Path(".cache/gcp_tiles"),
) -> AsyncGenerator[TileData, None]:
    """
    Fetches satellite tiles for a given bounding box and yields smaller split tiles.

    Args:
    session_token: The session token obtained from get_session_token function
    api_key: Your Google Maps Platform API key
    tile_size: The size of the original tiles
    start_lat: The latitude of the top-left corner of the bounding bbox
    start_lng: The longitude of the top-left corner of the bounding bbox
    end_lat: The latitude of the bottom-right corner of the bounding bbox
    end_lng: The longitude of the bottom-right corner of the bounding bbox
    zoom: The zoom level of the tiles
    target_tile_size: The size of the smaller tiles to split into
    cache_dir: The path to the cache directory

    Yields:
    Smaller split TileData objects
    """
    start_x, start_y = from_lat_lng_to_tile_coord(start_lat, start_lng, zoom, tile_size)
    end_x, end_y = from_lat_lng_to_tile_coord(end_lat, end_lng, zoom, tile_size)

    min_x, max_x = min(start_x, end_x), max(start_x, end_x)
    min_y, max_y = min(start_y, end_y), max(start_y, end_y)

    total_tiles = (max_x - min_x + 1) * (max_y - min_y + 1)
    logger.info(f"Fetching {total_tiles} tiles from GCP")

    progress_bar = tqdm(total=total_tiles, desc="Fetching tiles")

    async with aiohttp.ClientSession() as session:
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                tile_data = await get_satellite_tile(
                    session=session,
                    session_token=session_token,
                    api_key=api_key,
                    zoom=zoom,
                    x=x,
                    y=y,
                    cache_dir=cache_dir,
                )
                if tile_data:
                    split_tiles = split_satellite_tile(tile_data, target_tile_size)
                    for split_tile in split_tiles:
                        yield split_tile
                progress_bar.update(1)

    progress_bar.close()


def get_image_hash(image: Image.Image) -> str:
    """Generate a hash for an image based on its content and size."""
    img_array = np.array(image)
    return hashlib.md5(img_array.tobytes()).hexdigest()


def get_cache_path(image: Image.Image, model: str, cache_dir: Path) -> Path:
    """Generate a cache file path based on image characteristics and model."""
    img_hash = get_image_hash(image)
    width, height = image.size
    return cache_dir / model / f"{width}x{height}" / f"{img_hash}.npy"


async def get_embeddings(
    session: aiohttp.ClientSession,
    images: list[Image.Image],
    model: Literal["clip", "clay"],
    cache_dir: Path = Path(".cache/embeddings"),
) -> list[list[float]]:
    """Get embeddings for a batch of images using the bluesight.ai API asynchronously.

    Args:
    session: An aiohttp ClientSession object
    images: A list of PIL Image objects
    model: The name of the model to use for embeddings
    cache_dir: The directory to store cached embeddings

    Returns:
    A list of embeddings for each image
    """
    embeddings: list[list[float]] = []
    uncached_images: list[Image.Image] = []
    uncached_indices: list[int] = []

    for i, image in enumerate(images):
        cache_file = get_cache_path(image, model, cache_dir)
        if cache_file.exists():
            embedding: list[float] = np.load(cache_file).tolist()
            embeddings.append(embedding)
        else:
            uncached_images.append(image)
            uncached_indices.append(i)
            embeddings.append([])

    cached_count = len(images) - len(uncached_images)
    logger.info(
        f"{cached_count} out of {len(images)} embeddings are cached, fetching {len(uncached_images)} embeddings"
    )

    if uncached_images:
        url = "https://api.bluesight.ai/embeddings/img"

        payload = {"model": model, "images": []}
        for image in uncached_images:
            buffer = io.BytesIO()
            np.save(buffer, np.array(image, dtype=np.uint8))
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            payload["images"].append({"gsd": 0.6, "bands": ["red", "green", "blue"], "pixels": img_base64})
        headers = {"Content-Type": "application/json"}

        async with session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            new_embeddings: list[list[float]] = (await response.json())["embeddings"]

        for i, embedding in zip(uncached_indices, new_embeddings):
            cache_file = get_cache_path(images[i], model, cache_dir)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, np.array(embedding))
            embeddings[i] = embedding

    return embeddings


async def get_embeddings_text(
    session: aiohttp.ClientSession,
    texts: list[str],
) -> list[list[float]]:
    url = "https://api.bluesight.ai/embeddings/text"
    headers = {"Content-Type": "application/json"}
    async with session.post(url, json=texts, headers=headers) as response:
        response.raise_for_status()
        embeddings: list[list[float]] = (await response.json())["embeddings"]
    return embeddings


async def insert_to_postgres(
    conn: asyncpg.Connection,
    table_name: str,
    data: list[tuple[str, list[float]]],
) -> list[int]:
    """Insert a batch of data into a PostgreSQL table.

    Args:
    conn: An asyncpg Connection object
    table_name: The name of the table to insert the data into (must have location and embedding columns)
    data: A list of tuples containing the bounding box and embedding data

    Returns:
    A list of IDs for the inserted rows
    """
    query = f"""
    WITH input_rows(bbox, embedding) AS (
        SELECT * FROM unnest($1::text[], $2::text[])
    ), inserted AS (
        INSERT INTO {table_name} (location, embedding)
        SELECT ST_GeomFromText(input_rows.bbox, 4326)::geography, input_rows.embedding::vector
        FROM input_rows
        ON CONFLICT (location) DO NOTHING
        RETURNING id, location
    )
    SELECT COALESCE(i.id, e.id) AS id
    FROM input_rows
    LEFT JOIN inserted i ON ST_GeomFromText(input_rows.bbox, 4326)::geography = i.location
    LEFT JOIN {table_name} e ON ST_GeomFromText(input_rows.bbox, 4326)::geography = e.location
    """

    bboxes: list[str]
    embeddings: list[list[float]]
    bboxes, embeddings = zip(*data)

    embedding_strings = [f"[{','.join(map(str, emb))}]" for emb in embeddings]

    results = await conn.fetch(query, bboxes, embedding_strings)

    return [row["id"] for row in results]


async def insert_subtiles_to_postgres(
    conn: asyncpg.Connection,
    table_name: str,
    data: list[tuple[str, list[float], int]],
) -> list[int]:
    """Insert a batch of data into a PostgreSQL table.

    Args:
    conn: An asyncpg Connection object
    table_name: The name of the table to insert the data into (must have location, embedding, and parent_tile columns)
    data: A list of tuples containing the bounding box, embedding data, and parent_tile_id

    Returns:
    A list of IDs for the inserted rows
    """
    query = f"""
    WITH input_rows(bbox, embedding, parent_tile_id) AS (
        SELECT * FROM unnest($1::text[], $2::text[], $3::int[])
    )
    INSERT INTO {table_name} (location, embedding, parent_tile_id)
    SELECT
        ST_GeomFromText(bbox, 4326)::geography AS location,
        embedding::vector AS embedding,
        parent_tile_id
    FROM input_rows
    RETURNING id
    """

    bboxes: list[str]
    embeddings: list[list[float]]
    parent_tile_ids: list[int]
    bboxes, embeddings, parent_tile_ids = zip(*data)
    embedding_strings = [f"[{','.join(map(str, emb))}]" for emb in embeddings]

    results = await conn.fetch(query, bboxes, embedding_strings, parent_tile_ids)

    return [row["id"] for row in results]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def upload_to_supabase(
    supabase_client: AsyncClient,
    bucket_name: str,
    file_name: str,
    file_content: bytes,
) -> None:
    """Upload a file to Supabase storage asynchronously.

    Args:
    supabase_client: An asynchronous Supabase client object
    bucket_name: The name of the storage bucket
    file_name: The name of the file to be stored
    file_content: The content of the file as bytes
    """
    try:
        await supabase_client.storage.from_(bucket_name).upload(file_name, file_content, {"content-type": "image/png"})
        logger.debug(f"Uploaded file {file_name} to Supabase storage bucket {bucket_name}")
    except Exception as e:
        if eval(str(e)).get("error") != "Duplicate":
            logger.error(f"Error uploading file {file_name} to Supabase storage: {e}")
        else:
            logger.debug(f"File {file_name} already exists in Supabase storage bucket {bucket_name}")


async def insert_area(
    start_lat: float = 29.7624311358678,
    start_lng: float = -95.1304100547797,
    end_lat: float = 29.585439488973286,
    end_lng: float = -94.96849737842977,
    zoom: int = 18,
    scale: Literal["scaleFactor1x", "scaleFactor2x", "scaleFactor4x"] = "scaleFactor4x",
    target_tile_size: int = 256,
    embedding_model: Literal["clip", "clay"] = "clip",
    batch_size: int = 256,
    table_name: str = "clip_boxes_gcp_houston",
    bucket_name: str = "clip_boxes_gcp_houston",
):
    """Insert satellite tiles and embeddings for a specified area into a PostgreSQL database.

    SF: 37.811219311975265, -122.52665573314543, 37.69850383939589, -122.34837526369923
    LA port: 33.78120548898004, -118.32824973224359, 33.69450702401006, -118.13318898292924
    NY bay: 40.70484394607446, -74.19615310512826, 40.61944007176511, -73.97058948914278
    Houston port: 29.7624311358678, -95.1304100547797, 29.585439488973286, -94.96849737842977

    Args:
    start_lat: The latitude of the top-left corner of the bounding bbox
    start_lng: The longitude of the top-left corner of the bounding bbox
    end_lat: The latitude of the bottom-right corner of the bounding bbox
    end_lng: The longitude of the bottom-right corner of the bounding bbox
    zoom: The zoom level of the tiles
    scale: Scales-up the size of map elements
    target_tile_size: The size of the smaller tiles
    embedding_model: The name of the model to use for embeddings
    batch_size: The number of tiles to process and insert into the database at a time
    table_name: The name of the table to insert the data into
    bucket_name: The name of the Supabase storage bucket to upload the tiles to
    """

    GCP_API_KEY = os.getenv("GCP_API_KEY")
    if not GCP_API_KEY:
        raise ValueError("Please set the GCP_API_KEY environment variable")

    SUPABASE_POSTGRES_PASSWORD = os.getenv("SUPABASE_POSTGRES_PASSWORD")
    if not SUPABASE_POSTGRES_PASSWORD:
        raise ValueError("Please set the SUPABASE_POSTGRES_PASSWORD environment variable")
    postgres_uri = f"postgresql://postgres.biccczfztgnfaqzmizan:{urllib.parse.quote_plus(SUPABASE_POSTGRES_PASSWORD)}@aws-0-us-east-1.pooler.supabase.com:6543/postgres"

    SUPABASE_URL, SUPABASE_KEY = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Please set the SUPABASE_URL and SUPABASE_KEY environment variables")

    supabase_client: AsyncClient = await acreate_client(SUPABASE_URL, SUPABASE_KEY)

    tiles_cache_dir = Path(f".cache/gcp_tiles/{scale}")

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
    logger.info(f"Retrieving tiles with size {tile_size}, scale {scale}, zoom {zoom}, cache dir {tiles_cache_dir}")

    tiles_generator = fetch_tiles(
        session_token=session_token,
        api_key=GCP_API_KEY,
        tile_size=tile_size,
        start_lat=start_lat,
        start_lng=start_lng,
        end_lat=end_lat,
        end_lng=end_lng,
        zoom=zoom,
        target_tile_size=target_tile_size,
        cache_dir=tiles_cache_dir,
    )

    logger.info(f"Processing tiles and inserting into database in batches of {batch_size}")
    conn = await asyncpg.connect(postgres_uri, statement_cache_size=0)
    async with aiohttp.ClientSession() as session:
        batch: list[TileData] = []
        async for tile in tiles_generator:
            batch.append(tile)
            if len(batch) == batch_size:
                # Process and insert the batch
                batch_images = [tile["tile"] for tile in batch]
                batch_embeddings = await get_embeddings(session=session, images=batch_images, model=embedding_model)

                data = [
                    (bbox_to_wkt((tile["nw_lng"], tile["nw_lat"], tile["se_lng"], tile["se_lat"])), embedding)
                    for tile, embedding in zip(batch, batch_embeddings)
                ]

                tile_ids = await insert_to_postgres(conn=conn, table_name=table_name, data=data)

                upload_tasks = []
                for tile_id, tile_data in zip(tile_ids, batch):
                    img_byte_arr = io.BytesIO()
                    tile_data["tile"].save(img_byte_arr, format="PNG")
                    img_byte_arr = img_byte_arr.getvalue()

                    file_name = f"{tile_id}.png"
                    upload_task = upload_to_supabase(
                        supabase_client=supabase_client,
                        bucket_name=bucket_name,
                        file_name=file_name,
                        file_content=img_byte_arr,
                    )
                    upload_tasks.append(upload_task)

                await asyncio.gather(*upload_tasks)

                batch = []

        # Process any remaining tiles
        if batch:
            batch_images = [tile["tile"] for tile in batch]
            batch_embeddings = await get_embeddings(session=session, images=batch_images, model=embedding_model)

            data = [
                (bbox_to_wkt((tile["nw_lng"], tile["nw_lat"], tile["se_lng"], tile["se_lat"])), embedding)
                for tile, embedding in zip(batch, batch_embeddings)
            ]

            await insert_to_postgres(conn=conn, table_name=table_name, data=data)

    await conn.close()


if __name__ == "__main__":
    Fire(insert_area)
