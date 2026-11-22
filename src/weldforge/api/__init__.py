"""REST API for CAD plugins, the cell and the shop floor."""

from .store import JobRecord, Store, get_store

__all__ = ["JobRecord", "Store", "get_store", "create_app"]


def create_app(store: Store | None = None):
    """Build the FastAPI app. Imported lazily so the core needs no FastAPI."""
    from .app import create_app as _create_app

    return _create_app(store)
