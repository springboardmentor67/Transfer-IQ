"""Compatibility ASGI entrypoint.

This lets `uvicorn app:app` work from the project root by re-exporting
`app` from `api.main`.
"""

from api.main import app
