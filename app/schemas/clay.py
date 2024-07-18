from enum import Enum
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field, validator

from clay.config import metadata


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
    bands: List[str] = Field(examples=[["red", "green", "blue"]])
    pixels: List[List[List[float]]] = Field(
        examples=[
            [
                [[255.0, 0.0], [0.0, 255.0]],
                [[0.0, 255.0], [255.0, 0.0]],
                [[0.0, 0.0], [255.0, 255.0]],
            ]
        ],
        description="3D float array (number of bands, h, w) representing an image",
    )
    platform: str | None = Field(
        default=None,
        examples=list(metadata.keys()),
        description=f"One of {list(metadata.keys())}. Used to retrieve means, stds and wavelengths across bands. If not provided, means and stds will be calculated from list of images, and default wavelengths will be used",
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
                    pixels=[
                        [
                            [45.7, 82.1, 53.0, 120.4, 215.2, 64.5, 198.3, 150.6],
                            [205.9, 123.1, 97.2, 54.3, 189.4, 210.5, 78.6, 35.7],
                            [147.8, 213.9, 65.2, 142.6, 185.3, 99.4, 110.1, 250.0],
                            [34.6, 111.2, 200.5, 240.3, 90.9, 77.7, 165.8, 192.1],
                            [223.4, 87.5, 58.8, 176.3, 119.0, 48.2, 140.7, 99.9],
                            [203.2, 75.6, 132.4, 208.8, 61.5, 143.9, 80.1, 190.0],
                            [150.3, 172.5, 35.8, 55.6, 220.9, 88.0, 101.3, 115.7],
                            [93.4, 66.6, 123.7, 183.2, 158.4, 199.6, 44.1, 212.5],
                        ],
                        [
                            [45.7, 82.1, 53.0, 120.4, 215.2, 64.5, 198.3, 150.6],
                            [205.9, 123.1, 97.2, 54.3, 189.4, 210.5, 78.6, 35.7],
                            [147.8, 213.9, 65.2, 142.6, 185.3, 99.4, 110.1, 250.0],
                            [34.6, 111.2, 200.5, 240.3, 90.9, 77.7, 165.8, 192.1],
                            [223.4, 87.5, 58.8, 176.3, 119.0, 48.2, 140.7, 99.9],
                            [203.2, 75.6, 132.4, 208.8, 61.5, 143.9, 80.1, 190.0],
                            [150.3, 172.5, 35.8, 55.6, 220.9, 88.0, 101.3, 115.7],
                            [93.4, 66.6, 123.7, 183.2, 158.4, 199.6, 44.1, 212.5],
                        ],
                        [
                            [45.7, 82.1, 53.0, 120.4, 215.2, 64.5, 198.3, 150.6],
                            [205.9, 123.1, 97.2, 54.3, 189.4, 210.5, 78.6, 35.7],
                            [147.8, 213.9, 65.2, 142.6, 185.3, 99.4, 110.1, 250.0],
                            [34.6, 111.2, 200.5, 240.3, 90.9, 77.7, 165.8, 192.1],
                            [223.4, 87.5, 58.8, 176.3, 119.0, 48.2, 140.7, 99.9],
                            [203.2, 75.6, 132.4, 208.8, 61.5, 143.9, 80.1, 190.0],
                            [150.3, 172.5, 35.8, 55.6, 220.9, 88.0, 101.3, 115.7],
                            [93.4, 66.6, 123.7, 183.2, 158.4, 199.6, 44.1, 212.5],
                        ],
                    ],
                ),
                Image(
                    platform="naip",
                    bands=["red", "green", "blue"],
                    gsd=0.6,
                    point=(37.77625, -122.43267),
                    timestamp=1614422534,
                    pixels=[
                        [
                            [45.7, 82.1, 53.0, 120.4, 215.2, 64.5, 198.3, 150.6],
                            [205.9, 123.1, 97.2, 54.3, 189.4, 210.5, 78.6, 35.7],
                            [147.8, 213.9, 65.2, 142.6, 185.3, 99.4, 110.1, 250.0],
                            [34.6, 111.2, 200.5, 240.3, 90.9, 77.7, 165.8, 192.1],
                            [223.4, 87.5, 58.8, 176.3, 119.0, 48.2, 140.7, 99.9],
                            [203.2, 75.6, 132.4, 208.8, 61.5, 143.9, 80.1, 190.0],
                            [150.3, 172.5, 35.8, 55.6, 220.9, 88.0, 101.3, 115.7],
                            [93.4, 66.6, 123.7, 183.2, 158.4, 199.6, 44.1, 212.5],
                        ],
                        [
                            [45.7, 82.1, 53.0, 120.4, 215.2, 64.5, 198.3, 150.6],
                            [205.9, 123.1, 97.2, 54.3, 189.4, 210.5, 78.6, 35.7],
                            [147.8, 213.9, 65.2, 142.6, 185.3, 99.4, 110.1, 250.0],
                            [34.6, 111.2, 200.5, 240.3, 90.9, 77.7, 165.8, 192.1],
                            [223.4, 87.5, 58.8, 176.3, 119.0, 48.2, 140.7, 99.9],
                            [203.2, 75.6, 132.4, 208.8, 61.5, 143.9, 80.1, 190.0],
                            [150.3, 172.5, 35.8, 55.6, 220.9, 88.0, 101.3, 115.7],
                            [93.4, 66.6, 123.7, 183.2, 158.4, 199.6, 44.1, 212.5],
                        ],
                        [
                            [45.7, 82.1, 53.0, 120.4, 215.2, 64.5, 198.3, 150.6],
                            [205.9, 123.1, 97.2, 54.3, 189.4, 210.5, 78.6, 35.7],
                            [147.8, 213.9, 65.2, 142.6, 185.3, 99.4, 110.1, 250.0],
                            [34.6, 111.2, 200.5, 240.3, 90.9, 77.7, 165.8, 192.1],
                            [223.4, 87.5, 58.8, 176.3, 119.0, 48.2, 140.7, 99.9],
                            [203.2, 75.6, 132.4, 208.8, 61.5, 143.9, 80.1, 190.0],
                            [150.3, 172.5, 35.8, 55.6, 220.9, 88.0, 101.3, 115.7],
                            [93.4, 66.6, 123.7, 183.2, 158.4, 199.6, 44.1, 212.5],
                        ],
                    ],
                ),
            ]
        ]
    )


class Embeddings(BaseModel):
    embeddings: List[List[float]] = Field(
        examples=[[[228.0, 322.1], [234.0, 231.5]]], description="Embedding representing an area"
    )


class ClassificationLabels(BaseModel):
    labels: List[int] = Field(
        examples=[[0, 1, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]], description="Classification labels. Must start with 0."
    )


class ModelData(BaseModel):
    model_id: str


class TrainClassificationData(Images, ClassificationLabels):
    pass


class InferClassificationData(ModelData, Images):
    pass


class TrainResults(ModelData):
    train_details: Dict[str, Any] | None
