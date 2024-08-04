from typing import Any, Optional

from pydantic import BaseModel


class IPMetadataCreate(BaseModel):
    id: str
    """The IP address of the client that made the request."""

    data: Optional[dict[str, Any]] = None
    """The metadata associated with the IP address."""


class IPMetadata(IPMetadataCreate):
    created_at: int
    """The Unix timestamp (in seconds) for when the request was created."""
