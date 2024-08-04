from typing import Literal, Optional

from pydantic import BaseModel


class ModelMetadataCreate(BaseModel):
    task: Literal["classification", "segmentation"]
    """The task type, which can be either `classification` or `segmentation`."""


class ModelMetadata(ModelMetadataCreate):
    id: str
    """The model identifier, which can be referenced in the API endpoints."""

    created_at: int
    """The Unix timestamp (in seconds) for when the model was created."""

    bytes: Optional[int] = None
    """The size of the model, in bytes."""


class ModelMetadataUpdate(BaseModel):
    bytes: Optional[int] = None
    """The size of the model, in bytes."""
