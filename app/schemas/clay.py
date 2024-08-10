import random
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator

from clay.config import default_wavelengths, metadata


class Points(BaseModel):
    points: List[Tuple[float, float]] = Field(
        examples=[[(37.77625, -122.43267), (40.68926, -74.04457)]],
        description="List of 2D coordinates of points of interest",
    )
    size: int = Field(examples=[128, 64], description="Bounding box edge size in pixels. Should be dividable by 8.")


class Image(BaseModel):
    gsd: float = Field(
        examples=[0.6], description="gsd's of each band in images list (should be the same across all bands)"
    )
    bands: List[str] = Field(
        examples=[["red", "green", "blue"]],
        description="Order of bands in pixels array. Used to retrieve means, stds and wavelengths. Pick from 'blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir', 'nir08', 'swir16', 'swir22'.",
    )

    pixels: List[List[List[float]]] = Field(
        examples=[
            [[[random.uniform(0, 255) for _ in range(16)] for _ in range(16)] for _ in range(3)],
        ],
        description="3D float array (number of bands, h, w) representing an image.",
    )
    platform: str | None = Field(
        default=None,
        examples=list(metadata.keys()),
        description=f"One of {list(metadata.keys())}. Used to retrieve means, stds and wavelengths across bands. If not provided, means and stds will be calculated from list of images, and default wavelengths will be used",
    )
    wavelengths: List[float] | None = Field(
        default=None,
        examples=[[0.665, 0.56, 0.493]],
        description=f"Bands' wavelengths. Should be provided only if `platform` is `None`, because when `platform` is provided wavelengths will be loaded from config. If `platform` is `None` and wavelengths are not provided, default values from `sentinel-2-l2a` will be used: {default_wavelengths}",
    )
    point: Tuple[float, float] | None = Field(
        default=None,
        examples=[(37.77625, -122.43267), (40.68926, -74.04457)],
        description="(lat, lon) coordinate of the center of an image. Doesn't have a huge impact on model output.",
    )
    timestamp: int | None = Field(
        default=None,
        examples=[1714423534, 1614422534],
        description="Timestamp when image was taken. Doesn't have a huge impact on model output.",
    )

    @validator("platform")
    def check_platform_in_range(cls, value):
        if value is not None and value not in list(metadata.keys()):
            raise ValueError(f"Platform must be one of {list(metadata.keys())}")
        return value

    @validator("pixels")
    def validate_image_dimensions(cls, v):
        if not v:
            raise ValueError("Image cannot be empty")

        num_bands = len(v)
        height = len(v[0])
        width = len(v[0][0])

        if height != width:
            raise ValueError(f"Image must be square, got dimensions {height}x{width}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"Image dimensions must be divisible by 8, got {height}x{width}")

        for band in v[1:]:
            if len(band) != height or len(band[0]) != width:
                raise ValueError("All bands must have the same dimensions")

        return v


class Images(BaseModel):
    images: List[Image] = Field(
        examples=[
            [
                Image(
                    platform="naip",
                    bands=["red", "green", "blue"],
                    gsd=0.6,
                    point=(37.77625, -122.43267),
                    timestamp=1614422534,
                    pixels=[[[random.uniform(0, 255) for _ in range(16)] for _ in range(16)] for _ in range(3)],
                ),
                Image(
                    platform="naip",
                    bands=["red", "green", "blue"],
                    gsd=0.6,
                    point=(37.77625, -122.43267),
                    timestamp=1614422534,
                    pixels=[[[random.uniform(0, 255) for _ in range(16)] for _ in range(16)] for _ in range(3)],
                ),
            ]
        ]
    )


class EmbeddingsRequest(Images):
    model: Literal["clay", "clip"] = Field(default="clay", description="Model to use for embeddings generation")


class Embeddings(BaseModel):
    embeddings: List[List[float]] = Field(
        examples=[[[228.0, 322.1], [234.0, 231.5]]], description="Embedding representing an area"
    )


class ClassificationTrainingDataSample(BaseModel):
    image: Image
    label: int


class SegmentationTrainingDataSample(BaseModel):
    image: Image
    label: List[List[int]]


class InferenceData(BaseModel):
    model: str
    images: List[Image]


class ClassificationLabels(BaseModel):
    labels: List[int]


class SegmentationLabels(BaseModel):
    labels: List[List[List[int]]] = Field(
        examples=[[[[random.randint(0, 1) for _ in range(16)] for _ in range(16)] for _ in range(2)]],
        description="3D (b, h, w) label array.",
    )
