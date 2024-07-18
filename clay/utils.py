import math
from datetime import datetime
from typing import Any, Dict, List, Tuple, TypedDict

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

from clay.config import config, default_wavelengths, device, metadata


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


def get_bbox(lat: float, lon: float, size: int, gsd: int) -> Bbox:
    bounds = (
        lat + shift_latitude(pixel_shift=int(0.5 * size), gsd=gsd),
        lon + shift_longitude(lat=lat, pixel_shift=int(-0.5 * size), gsd=gsd),
        lat + shift_latitude(pixel_shift=int(-0.5 * size), gsd=gsd),
        lon + shift_longitude(lat=lat, pixel_shift=int(0.5 * size), gsd=gsd),
    )
    return bounds


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
    y, x = y.to(device), x.to(device)
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = omega.to(device)
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
    omega = omega.to(device)
    pos = pos.to(device)

    scaled_pos = pos[:, None] * omega[None, :]
    pe = torch.cat((scaled_pos.sin(), scaled_pos.cos()), dim=1)

    return pe.type(dtype)


# Prep datetimes embedding using a normalization function from the model code.
def normalize_timestamp(date: datetime):
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

    logger.debug(
        f"Searching catalogue for {max_items} item(s) at ({lat}, {lon}) from {start} to {end} with {bb_offset} offset..."
    )

    # Search the catalogue
    catalog = pystac_client.Client.open(config.stac_api_url)
    search = catalog.search(
        collections=[config.platform],
        datetime=f"{start}/{end}",
        bbox=(lon - bb_offset, lat - bb_offset, lon + bb_offset, lat + bb_offset),
        max_items=max_items,
        query={"eo:cloud_cover": {"lt": 5}} if config.platform != "naip" else None,
        sortby="properties.eo:cloud_cover" if config.platform != "naip" else None,
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
        raise ValueError(f"Unable to find any items at ({lat}, {lon}) from {start} to {end} with {bb_offset} offset!")

    return items


def get_bounds(lat: float, lon: float, epsg: int, gsd: float, size: int) -> Bbox:

    logger.debug(f"Calculating bounds for ({lat}, {lon}) with epsg={epsg}, gsd={gsd}, and size={size}...")

    # Convert point of interest into the image projection
    # (assumes all images are in the same projection)
    poidf = gpd.GeoDataFrame(
        pd.DataFrame(),
        crs="EPSG:4326",
        geometry=[Point(lon, lat)],
    ).to_crs(epsg)

    if poidf is None or poidf.empty:
        raise ValueError("DataFrame is empty!")

    coords = poidf.iloc[0].geometry.coords[0]

    # Create bounds in projection
    bounds: Bbox = (
        coords[0] - (size * gsd) / 2,
        coords[1] - (size * gsd) / 2,
        coords[0] + (size * gsd) / 2,
        coords[1] + (size * gsd) / 2,
    )

    return bounds


def get_stack(lat: float, lon: float, items: List[pystac.Item], size: int, gsd: float) -> DataArray:

    # Extract coordinate system from first item
    epsg: int = items[0].properties["proj:epsg"]

    bounds = get_bounds(lat=lat, lon=lon, epsg=epsg, gsd=gsd, size=size)

    logger.debug(bounds)

    match config.platform:
        case "naip":

            import rioxarray
            import xarray as xr

            item = items[0]

            dataset = rioxarray.open_rasterio(item.assets["image"].href).sel(band=[1, 2, 3, 4])
            granule_name = item.assets["image"].href.split("/")[-1]

            # dataset = dataset.transpose("band", "y", "x")
            # dataset = dataset.isel(x=slice(1, -1), y=slice(1, -1))

            tile = dataset.rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3])
            tile = tile.assign_coords(band=["red", "green", "blue", "nir"])
            # logger.info(tile)

            current_shape = tile.shape[1:]  # (height, width)
            if current_shape[0] > size or current_shape[1] > size:
                logger.warning(f"Current shape {current_shape} is bigger than target shape ({size}, {size})!")
                x_center = current_shape[1] // 2
                y_center = current_shape[0] // 2
                x_start = max(0, x_center - size // 2)
                y_start = max(0, y_center - size // 2)
                x_end = x_start + size
                y_end = y_start + size
                tile = tile.isel(x=slice(x_start, x_end), y=slice(y_start, y_end))

            if current_shape[0] < size or current_shape[1] < size:
                raise ValueError(f"Current shape {current_shape} is smaller than target shape ({size}, {size})!")

            time_coord = xr.DataArray([item.properties["datetime"]], dims="time", name="time")
            tile = tile.expand_dims(time=[0])
            tile = tile.assign_coords(time=time_coord)

            gsd_coord = xr.DataArray([gsd], dims="gsd", name="gsd")
            tile = tile.expand_dims(gsd=[0])
            tile = tile.assign_coords(gsd=gsd_coord)
            tile = tile[0]  # gsd
            stack = tile
        case _:
            stack: DataArray = stackstac.stack(
                items,
                bounds=bounds,
                snap_bounds=False,
                epsg=epsg,
                resolution=gsd,
                dtype=np.dtype("float32"),
                rescale=False,
                fill_value=0,
                assets=config.assets,
                resampling=Resampling.nearest,
            )
    stack = stack.compute()

    return stack


def visualize_stack(stack, figsize=(40, 40)):
    stack.sel(band=["red", "green", "blue"]).plot.imshow(
        row="time", rgb="band", vmin=0, vmax=2000, col_wrap=6, figsize=figsize
    )
    for ax in plt.gcf().axes:
        ax.set_aspect("equal")
    plt.show()


def stack_to_datacube(lat: float, lon: float, stack: DataArray) -> Dict[str, Any]:

    mean = []
    std = []
    waves = []
    for band in stack.band:
        mean.append(metadata[config.platform]["bands"]["mean"][str(band.values)])
        std.append(metadata[config.platform]["bands"]["std"][str(band.values)])
        waves.append(metadata[config.platform]["bands"]["wavelength"][str(band.values)])

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
        "platform": config.platform,
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


class Stats(TypedDict):
    mean: List[float]
    std: List[float]
    waves: List[float]


def get_stats(
    bands: List[str], pixels: List[List[List[List[float]]]], platform: str | None, wavelengths: List[float] | None
) -> Stats:
    mean: List[float] = []
    std: List[float] = []
    waves: List[float] = []

    if platform:
        for band in bands:
            mean.append(metadata[platform]["bands"]["mean"][band])
            std.append(metadata[platform]["bands"]["std"][band])
            waves.append(metadata[platform]["bands"]["wavelength"][band])
    else:
        wavelengths_final = dict(zip(bands, wavelengths)) if wavelengths else default_wavelengths
        pixels_array = np.array(pixels)
        for i, band in enumerate(bands):
            band_data = pixels_array[:, i, :, :].flatten()
            mean.append(float(np.mean(band_data)))
            std.append(float(np.std(band_data)))
            waves.append(wavelengths_final.get(band, 0.0))

    return Stats(mean=mean, std=std, waves=waves)


def get_datacube(
    gsd: float,
    stats: Stats,
    pixels: List[List[List[List[float]]]],
    points: List[Tuple[float, float] | None],
    datetimes: List[datetime | None],
):

    # Prepare the normalization transform function using the mean and std values.
    transform = v2.Compose(
        [
            v2.Normalize(mean=stats["mean"], std=stats["std"]),
        ]
    )

    times = []
    for date in datetimes:
        if date is None:
            times.append(torch.zeros(4, dtype=torch.float32, device=device))
        else:
            normalized = normalize_timestamp(date)
            times.append(torch.tensor(np.hstack((normalized[0], normalized[1])), dtype=torch.float32, device=device))
    time = torch.stack(times)

    latlons = []
    for point in points:
        if point is None or None in point:
            latlons.append(torch.zeros(4, dtype=torch.float32, device=device))
        else:
            lat, lon = point
            normalized = normalize_latlon(lat, lon)
            latlons.append(torch.tensor(np.hstack((normalized[0], normalized[1])), dtype=torch.float32, device=device))
    latlon = torch.stack(latlons)

    # Normalize pixels
    pixels_numpy = np.array(pixels, dtype=np.float32)
    pixels_torch = torch.from_numpy(pixels_numpy)
    pixels_torch = transform(pixels_torch)

    datacube: Dict[str, Any] = {
        "platform": config.platform,
        "time": time,
        "latlon": latlon,
        "pixels": pixels_torch.to(device),
        "gsd": torch.tensor(gsd, device=device),
        # "gsd": torch.tensor([10], device=device),
        "waves": torch.tensor(stats["waves"], device=device),
    }
    return datacube
