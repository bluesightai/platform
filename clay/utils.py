"""
Code from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

"""

import math
from typing import Any, Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac_client
import stackstac
import torch
from pydantic import BaseModel
from rasterio.enums import Resampling
from shapely import Point
from stackstac.stack import Bbox
from torchvision.transforms import v2
from xarray import DataArray

from clay.args import args, device, metadata


class InputEarthData(BaseModel):
    items: List[Any]
    bounds: Bbox
    snap_bounds: bool
    epsg: int
    resolution: int
    rescale: bool
    fill_value: Union[int, float]
    assets: List[str]
    resampling: Resampling
    lat: float
    long: float


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


def get_mock_data() -> Tuple[Tuple[DataArray, float, float], InputEarthData]:
    # Point over Monchique Portugal
    lat, lon = 37.30939, -8.57207

    # Dates of a large forest fire
    start = "2018-07-01"
    end = "2018-09-01"

    STAC_API = "https://earth-search.aws.element84.com/v1"
    COLLECTION = "sentinel-2-l2a"

    # Search the catalogue
    catalog = pystac_client.Client.open(STAC_API)
    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{start}/{end}",
        bbox=(lon - 1e-5, lat - 1e-5, lon + 1e-5, lat + 1e-5),
        max_items=100,
        query={"eo:cloud_cover": {"lt": 80}},
    )

    all_items = search.get_all_items()

    # Reduce to one per date (there might be some duplicates
    # based on the location)
    items = []
    dates = []
    for item in all_items:
        if item.datetime.date() not in dates:
            items.append(item)
            dates.append(item.datetime.date())

    # Extract coordinate system from first item
    epsg = items[0].properties["proj:epsg"]

    # Convert point of interest into the image projection
    # (assumes all images are in the same projection)
    poidf = gpd.GeoDataFrame(
        pd.DataFrame(),
        crs="EPSG:4326",
        geometry=[Point(lon, lat)],
    ).to_crs(epsg)

    coords = poidf.iloc[0].geometry.coords[0]

    # Create bounds in projection
    size = 256
    gsd = 10
    bounds: Bbox = (
        coords[0] - (size * gsd) // 2,
        coords[1] - (size * gsd) // 2,
        coords[0] + (size * gsd) // 2,
        coords[1] + (size * gsd) // 2,
    )

    stack = stackstac.stack(
        items,
        bounds=bounds,
        snap_bounds=False,
        epsg=epsg,
        resolution=gsd,
        # dtype=np.dtype("float32"),
        rescale=False,
        fill_value=0,
        assets=["blue", "green", "red", "nir"],
        resampling=Resampling.nearest,
    )
    input_earth_data: InputEarthData = InputEarthData(
        items=items,
        bounds=bounds,
        snap_bounds=False,
        epsg=epsg,
        resolution=gsd,
        # dtype=np.dtype("float32"),
        rescale=False,
        fill_value=0,
        assets=["blue", "green", "red", "nir"],
        resampling=Resampling.nearest,
        lat=lat,
        long=lon,
    )

    stack = stack.compute()

    return (stack, lat, lon), input_earth_data


def stack_to_datacube(stack: DataArray, lat: float, long: float) -> Dict[str, Any]:

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

    latlons = [normalize_latlon(lat, long)] * len(times)
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
        "waves": torch.tensor(waves, device=device),
    }

    return datacube
