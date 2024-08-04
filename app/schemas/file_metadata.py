from typing import Optional

from pydantic import BaseModel


class FileMetadataCreate(BaseModel):
    filename: Optional[str] = None
    """The name of the file."""

    bytes: Optional[int] = None
    """The size of the file, in bytes."""


class FileMetadata(FileMetadataCreate):
    id: str
    """The file identifier, which can be referenced in the API endpoints."""

    created_at: int
    """The Unix timestamp (in seconds) for when the file was created."""


class FileMetadataUpdate(BaseModel):
    bytes: Optional[int] = None
    """The size of the file, in bytes."""
