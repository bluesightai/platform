from typing import Optional

from pydantic import BaseModel, ConfigDict


class FileMetadataCreate(BaseModel):
    filename: Optional[str] = None
    """The name of the file."""

    bytes: Optional[int] = None
    """The size of the file, in bytes."""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        json_schema_extra={
            "examples": [
                {
                    "filename": "forest_fire_train_data.h5",
                    "bytes": 3669052,
                }
            ]
        },
    )


class FileMetadata(FileMetadataCreate):
    id: str
    """The file identifier, which can be referenced in the API endpoints."""

    created_at: int
    """The Unix timestamp (in seconds) for when the file was created."""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        json_schema_extra={
            "examples": [
                {
                    "filename": "forest_fire_train_data.h5",
                    "bytes": 3669052,
                    "id": "file-lw3zjxrg",
                    "created_at": 1722451315,
                }
            ]
        },
    )


class FileMetadataUpdate(BaseModel):
    bytes: Optional[int] = None
    """The size of the file, in bytes."""
