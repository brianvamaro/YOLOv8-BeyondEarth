"""Smoke tests: confirm the package imports under the configured env + path wiring.

These are intentionally trivial. They exist so `pytest` collection proves the test
harness, the `src/` layout, and the `bouldernet` env are wired up correctly before any
real tests are added.
"""


def test_package_imports():
    import YOLOv8BeyondEarth  # noqa: F401


def test_predict_module_imports():
    from YOLOv8BeyondEarth import predict  # noqa: F401

    assert hasattr(predict, "get_sliced_prediction")
