from typing import List, Tuple

from pydantic import BaseModel, Field


class Points(BaseModel):
    points: List[Tuple[float, float]] = Field(
        examples=[[(37.77625, -122.43267), (40.68926, -74.04457)]],
        description="List of 2D coordinates of points of interest",
    )


class Embeddings(BaseModel):
    embeddings: List[List[float]] = Field(
        examples=[[[228.0, 322.1], [234.0, 231.5]]], description="Embedding representing an area"
    )


class ImageData(BaseModel):
    image: List[List[List[float]]] = Field(
        examples=[
            [
                [[255.0, 0.0], [0.0, 255.0]],
                [[0.0, 255.0], [255.0, 0.0]],
                [[0.0, 0.0], [255.0, 255.0]],
            ]
        ],
        description="3D float array with dimensions [x (number of bands), h, w] representing an image",
    )
