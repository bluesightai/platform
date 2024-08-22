import math
import os
from io import BytesIO
from pathlib import Path
from pprint import pprint
from typing import List, Literal, Optional, Tuple, TypedDict, Union

import matplotlib.pyplot as plt
import requests
from fire import Fire
from PIL import Image


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


def get_session_token(
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
    api_key (str): Your Google Maps Platform API key
    map_type (MapType): The type of base map
    language (str): IETF language tag
    region (str): CLDR region identifier
    image_format (Optional[ImageFormat]): Specifies the file format to return
    scale (ScaleFactor): Scales-up the size of map elements
    high_dpi (bool): Specifies whether to return high-resolution tiles
    layer_types (Optional[List[LayerType]]): Array of layer types to add to the map
    styles (Optional[List[StyleDict]]): Array of JSON style objects for map customization
    overlay (Optional[bool]): Specifies whether layerTypes should be rendered as a separate overlay
    api_options (Optional[List[str]]): Array of additional API options

    Returns:
    Optional[SessionResponse]: A dictionary containing the session token and related information, or None if an error occurs
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
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving session token: {e}")
        return None


def from_lat_lng_to_point(lat: float, lng: float, tile_size: int) -> Tuple[float, float]:
    """Convert latitude and longitude to point coordinates."""
    siny = math.sin((lat * math.pi) / 180)
    siny = min(max(siny, -0.9999), 0.9999)

    x = tile_size * (0.5 + lng / 360)
    y = tile_size * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))

    return (x, y)


def from_lat_lng_to_tile_coord(lat: float, lng: float, zoom: int, tile_size: int) -> Tuple[int, int, int]:
    """Convert latitude and longitude to tile coordinates."""
    x, y = from_lat_lng_to_point(lat, lng, tile_size)
    scale = 1 << zoom

    return (int(x * scale / tile_size), int(y * scale / tile_size), zoom)


def get_satellite_tile(
    session_token: str, api_key: str, zoom: int, x: int, y: int, output_path: Optional[Path] = None
) -> Optional[bytes]:
    """
    Retrieves a satellite tile from the Google Maps Platform Map Tiles API.

    Args:
    session_token (str): The session token obtained from get_session_token function
    api_key (str): Your Google Maps Platform API key
    zoom (int): The zoom level of the tile (typically 0-22, depends on the area)
    x (int): The x coordinate of the tile
    y (int): The y coordinate of the tile
    output_path (Optional[Path]): If provided, saves the tile to this file path

    Returns:
    Optional[bytes]: The tile image data as bytes if successful, None if an error occurs
    """
    url = f"https://tile.googleapis.com/v1/2dtiles/{zoom}/{x}/{y}"

    params = {"session": session_token, "key": api_key}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        if output_path:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Tile saved to {output_path}")

        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving satellite tile: {e}")
        return None


if __name__ == "__main__":
    lat, lng = 37.77979328255082, -122.42275579152276
    zoom = 21

    GCP_API_KEY = os.getenv("GCP_API_KEY")
    if not GCP_API_KEY:
        raise ValueError("Please set the GCP_API_KEY environment variable")
    session_data = get_session_token(
        api_key=GCP_API_KEY,
        map_type="satellite",
        language="en-US",
        region="US",
        image_format="png",
        scale="scaleFactor4x",
        high_dpi=True,
        # layer_types=["layerRoadmap", "layerStreetview"],
        # styles=[{"stylers": [{"hue": "#00ffe6"}, {"saturation": -20}]}],
        # overlay=True,
        # api_options=["MCYJ5E517XR2JC"],
    )
    if not session_data:
        raise ValueError("Error retrieving session data")
    pprint(session_data)
    session_token = session_data.get("session")
    tile_size = session_data.get("tileWidth")

    x, y, _ = from_lat_lng_to_tile_coord(lat, lng, zoom, tile_size)
    print(f"Tile coordinates: x={x}, y={y}, zoom={zoom}")

    tile_data = get_satellite_tile(
        session_token=session_token,
        api_key=GCP_API_KEY,
        zoom=zoom,
        x=x,
        y=y,
        output_path=Path("sf_satellite.png"),
    )
    if not tile_data:
        raise ValueError("Error retrieving tile data")

    plt.imshow(Image.open(BytesIO(tile_data)))
    plt.axis("off")
    plt.show()
