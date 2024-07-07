from typing import List

import torch
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """These values may be overriden by envs"""

    hub_repo_id: str = "made-with-clay/Clay"
    hub_filename: str = "clay-v1-base.ckpt"
    platform: str = "naip"
    stac_api_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    # platform: str = "sentinel-2-l2a"
    # stac_api_url: str = "https://earth-search.aws.element84.com/v1"

    assets: List[str] = [
        "blue",
        "green",
        "red",
        "nir",
        # "rededge1",
        # "rededge2",
        # "rededge3",
        # "nir08",
        # "swir16",
        # "swir32",
    ]


config = Config()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
metadata = {
    "sentinel-2-l2a": {
        "band_order": ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"],
        "rgb_indices": [2, 1, 0],
        "gsd": 10,
        "bands": {
            "mean": {
                "blue": 1105.0,
                "green": 1355.0,
                "red": 1552.0,
                "rededge1": 1887.0,
                "rededge2": 2422.0,
                "rededge3": 2630.0,
                "nir": 2743.0,
                "nir08": 2785.0,
                "swir16": 2388.0,
                "swir22": 1835.0,
            },
            "std": {
                "blue": 1809.0,
                "green": 1757.0,
                "red": 1888.0,
                "rededge1": 1870.0,
                "rededge2": 1732.0,
                "rededge3": 1697.0,
                "nir": 1742.0,
                "nir08": 1648.0,
                "swir16": 1470.0,
                "swir22": 1379.0,
            },
            "wavelength": {
                "blue": 0.493,
                "green": 0.56,
                "red": 0.665,
                "rededge1": 0.704,
                "rededge2": 0.74,
                "rededge3": 0.783,
                "nir": 0.842,
                "nir08": 0.865,
                "swir16": 1.61,
                "swir22": 2.19,
            },
        },
    },
    "landsat-c2l1": {
        "band_order": ["red", "green", "blue", "nir08", "swir16", "swir22"],
        "rgb_indices": [0, 1, 2],
        "gsd": 30,
        "bands": {
            "mean": {
                "red": 10678.0,
                "green": 10563.0,
                "blue": 11083.0,
                "nir08": 14792.0,
                "swir16": 12276.0,
                "swir22": 10114.0,
            },
            "std": {
                "red": 6025.0,
                "green": 5411.0,
                "blue": 5468.0,
                "nir08": 6746.0,
                "swir16": 5897.0,
                "swir22": 4850.0,
            },
            "wavelength": {"red": 0.65, "green": 0.56, "blue": 0.48, "nir08": 0.86, "swir16": 1.6, "swir22": 2.2},
        },
    },
    "landsat-c2l2-sr": {
        "band_order": ["red", "green", "blue", "nir08", "swir16", "swir22"],
        "rgb_indices": [0, 1, 2],
        "gsd": 30,
        "bands": {
            "mean": {
                "red": 13705.0,
                "green": 13310.0,
                "blue": 12474.0,
                "nir08": 17801.0,
                "swir16": 14615.0,
                "swir22": 12701.0,
            },
            "std": {
                "red": 9578.0,
                "green": 9408.0,
                "blue": 10144.0,
                "nir08": 8277.0,
                "swir16": 5300.0,
                "swir22": 4522.0,
            },
            "wavelength": {"red": 0.65, "green": 0.56, "blue": 0.48, "nir08": 0.86, "swir16": 1.6, "swir22": 2.2},
        },
    },
    "naip": {
        "band_order": ["red", "green", "blue", "nir"],
        "rgb_indices": [0, 1, 2],
        "gsd": 1.0,
        "bands": {
            "mean": {"red": 110.16, "green": 115.41, "blue": 98.15, "nir": 139.04},
            "std": {"red": 47.23, "green": 39.82, "blue": 35.43, "nir": 49.86},
            "wavelength": {"red": 0.65, "green": 0.56, "blue": 0.48, "nir": 0.842},
        },
    },
    "linz": {
        "band_order": ["red", "green", "blue"],
        "rgb_indices": [0, 1, 2],
        "gsd": 0.5,
        "bands": {
            "mean": {"red": 89.96, "green": 99.46, "blue": 89.51},
            "std": {"red": 41.83, "green": 36.96, "blue": 31.45},
            "wavelength": {"red": 0.635, "green": 0.555, "blue": 0.465},
        },
    },
    "sentinel-1-rtc": {
        "band_order": ["vv", "vh"],
        "gsd": 10,
        "bands": {
            "mean": {"vv": 0.123273, "vh": 0.027337},
            "std": {"vv": 1.492154, "vh": 0.122182},
            "wavelength": {"vv": 3.5, "vh": 4.0},
        },
    },
}
