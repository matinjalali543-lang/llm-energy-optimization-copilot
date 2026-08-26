"""Shared fixtures.

Matplotlib is pinned to the Agg backend **here and only here**. An earlier repo
in this series called `matplotlib.use("Agg")` inside a library module, and every
notebook figure rendered as the string `<Figure size 640x480>` instead of a
picture. The backend belongs to whatever runs the code; a test suite may declare
it and a library never may.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from energy_copilot.domain.config import SiteConfig  # noqa: E402


@pytest.fixture(scope="session")
def site() -> SiteConfig:
    return SiteConfig()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
