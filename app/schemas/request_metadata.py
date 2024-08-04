from typing import Any, Optional

from pydantic import BaseModel


class RequestMetadataCreate(BaseModel):
    ip: str
    """The IP address of the client that made the request."""

    headers: dict[str, Any]
    """The headers of the request."""

    method: str
    """The HTTP method of the request."""

    url: str
    """The URL of the request."""

    query_params: dict[str, Any]
    """The query parameters of the request."""

    body: Optional[dict[str, Any]] = None
    """The body of the request."""

    response_status_code: int
    """The status code of the response."""

    response: dict[str, Any]
    """The body of the response."""

    process_time: float
    """The time taken to process the request, in seconds."""


class RequestMetadata(RequestMetadataCreate):
    id: str
    """The unique identifier of the request."""

    created_at: int
    """The Unix timestamp (in seconds) for when the request was created."""
