import math
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystac
import pystac_client
import stackstac
import torch
from loguru import logger
from rasterio.enums import Resampling
from shapely import Point
from stackstac.stack import Bbox
from torchvision.transforms import v2
from xarray import DataArray

from clay.args import args, device, metadata


def shift_latitude(pixel_shift: int, gsd: int) -> float:
    """
    Calculate the latitude shift for a given pixel shift and GSD.

    Parameters:
    pixel_shift: Number of pixels to shift.
    gsd: Ground Sample Distance in meters/pixel.

    Returns:
    Shift in latitude in degrees.
    """
    shift_meters = pixel_shift * gsd
    latitude_shift = shift_meters / 111320
    return latitude_shift


def shift_longitude(lat: float, pixel_shift: int, gsd: int) -> float:
    """
    Calculate the longitude shift for a given pixel shift, GSD, and latitude.

    Parameters:
    lat: Latitude of the point.
    pixel_shift: Number of pixels to shift.
    gsd: Ground Sample Distance in meters/pixel.

    Returns:
    Shift in longitude in degrees.
    """
    lat_rad = math.radians(lat)
    shift_meters = pixel_shift * gsd
    longitude_shift = shift_meters / (111320 * math.cos(lat_rad))
    return longitude_shift


def get_square_centers(
    nw_lat: float, nw_lon: float, se_lat: float, se_lon: float, pixel_shift: int, gsd: int
) -> List[Tuple[float, float]]:
    centers = []
    current_lat = nw_lat
    latitude_shift = shift_latitude(pixel_shift, gsd)
    while current_lat >= se_lat:
        current_lon = nw_lon
        while current_lon <= se_lon:
            centers.append((current_lat, current_lon))
            current_lon += shift_longitude(current_lat, pixel_shift, gsd)
        current_lat -= latitude_shift
    return centers


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def find_closest_point(centers: List[Tuple[float, float]], target_lat: float, target_lon: float) -> int:
    closest_point_idx = 0
    min_distance = float("inf")

    for i, center in enumerate(centers):
        center_lat, center_lon = center
        distance = haversine_distance(center_lat, center_lon, target_lat, target_lon)

        if distance < min_distance:
            min_distance = distance
            closest_point_idx = i

    return closest_point_idx


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_2d_with_gsd(h, w, dim, gsd=1.0, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** (2 * omega / dim)) * (gsd / 1.0)  # Adjusted for g

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_1d(pos, dim, temperature: int = 10000, dtype=torch.float32):
    assert dim % 2 == 0, "Feature dimension must be a multiple of 2 for sincos embedding"
    pos = torch.arange(pos) if isinstance(pos, int) else pos

    omega = torch.arange(dim // 2) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    scaled_pos = pos[:, None] * omega[None, :]
    pe = torch.cat((scaled_pos.sin(), scaled_pos.cos()), dim=1)

    return pe.type(dtype)


# Prep datetimes embedding using a normalization function from the model code.
def normalize_timestamp(date):
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24

    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


# Prep lat/lon embedding using the
def normalize_latlon(lat, lon):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))


def get_catalog_items(
    lat: float,
    lon: float,
    start: str = "2024-01-01",
    end: str = "2024-05-01",
    bb_offset: float = 1e-5,
    max_items: int = 1,
) -> List[pystac.Item]:

    logger.info(
        f"Searching catalogue for {max_items} item(s) at ({lat}, {lon}) from {start} to {end} with {bb_offset} offset..."
    )

    # Search the catalogue
    catalog = pystac_client.Client.open(args.stac_api_url)
    search = catalog.search(
        collections=[args.platform],
        datetime=f"{start}/{end}",
        bbox=(lon - bb_offset, lat - bb_offset, lon + bb_offset, lat + bb_offset),
        max_items=max_items,
        query={"eo:cloud_cover": {"lt": 5}},
        sortby="properties.eo:cloud_cover",
    )

    all_items = search.item_collection()

    # Reduce to one per date (there might be some duplicates
    # based on the location)
    items: List[pystac.Item] = []
    dates = set()
    for item in all_items:
        if item.datetime and item.datetime.date() not in dates:
            items.append(item)
            dates.add(item.datetime.date())

    if not items:
        raise ValueError("Unable to find any items at ({lat}, {lon}) from {start} to {end} with {bb_offset} offset!")

    return items


def get_bounds(lat: float, lon: float, epsg: int, gsd: int = 10, size: int = 64) -> Bbox:

    # Convert point of interest into the image projection
    # (assumes all images are in the same projection)
    poidf = gpd.GeoDataFrame(
        pd.DataFrame(),
        crs="EPSG:4326",
        geometry=[Point(lon, lat)],
    ).to_crs(epsg)

    if poidf is None:
        raise ValueError("DataFrame is empty!")

    coords = poidf.iloc[0].geometry.coords[0]

    # Create bounds in projection
    bounds: Bbox = (
        coords[0] - (size * gsd) // 2,
        coords[1] - (size * gsd) // 2,
        coords[0] + (size * gsd) // 2,
        coords[1] + (size * gsd) // 2,
    )

    return bounds


def get_stack(lat: float, lon: float, items: List[pystac.Item], size: int = 64, gsd: int = 10) -> DataArray:

    # Extract coordinate system from first item
    epsg: int = items[0].properties["proj:epsg"]

    bounds = get_bounds(lat=lat, lon=lon, epsg=epsg, gsd=gsd, size=size)

    stack: DataArray = stackstac.stack(
        items,
        bounds=bounds,
        snap_bounds=False,
        epsg=epsg,
        resolution=gsd,
        dtype=np.dtype("float32"),
        rescale=False,
        fill_value=0,
        assets=args.assets,
        resampling=Resampling.nearest,
    )
    stack = stack.compute()

    return stack


def visualize_stack(stack):
    stack.sel(band=["red", "green", "blue"]).plot.imshow(row="time", rgb="band", vmin=0, vmax=2000, col_wrap=6)
    for ax in plt.gcf().axes:
        ax.set_aspect("equal")
    plt.show()


def stack_to_datacube(lat: float, lon: float, stack: DataArray) -> Dict[str, Any]:

    mean = []
    std = []
    waves = []
    for band in stack.band:
        mean.append(metadata[args.platform]["bands"]["mean"][str(band.values)])
        std.append(metadata[args.platform]["bands"]["std"][str(band.values)])
        waves.append(metadata[args.platform]["bands"]["wavelength"][str(band.values)])

    # Prepare the normalization transform function using the mean and std values.
    transform = v2.Compose(
        [
            v2.Normalize(mean=mean, std=std),
        ]
    )

    datetimes = stack.time.values.astype("datetime64[s]").tolist()
    times = [normalize_timestamp(dat) for dat in datetimes]
    week_norm = [dat[0] for dat in times]
    hour_norm = [dat[1] for dat in times]

    latlons = [normalize_latlon(lat, lon)] * len(times)
    lat_norm = [dat[0] for dat in latlons]
    lon_norm = [dat[1] for dat in latlons]

    # Normalize pixels
    pixels = torch.from_numpy(stack.data.astype(np.float32))
    pixels = transform(pixels)

    datacube: Dict[str, Any] = {
        "platform": args.platform,
        "time": torch.tensor(
            np.hstack((week_norm, hour_norm)),
            dtype=torch.float32,
            device=device,
        ),
        "latlon": torch.tensor(np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=device),
        "pixels": pixels.to(device),
        "gsd": torch.tensor(stack.gsd.values, device=device),
        # "gsd": torch.tensor([10], device=device),
        "waves": torch.tensor(waves, device=device),
    }

    return datacube
