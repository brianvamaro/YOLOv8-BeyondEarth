"""Shared pytest fixtures and configuration for the YOLOv8BeyondEarth test suite.

The package lives under ``src/`` (src layout); ``pythonpath = ["src"]`` in
``pyproject.toml`` puts it on the import path, so tests can ``import YOLOv8BeyondEarth``
without an editable install.

Add shared fixtures here as the suite grows (e.g. sample polygons, a tiny test raster,
known-angle synthetic ellipses for the orientation characterization test).
"""
