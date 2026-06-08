"""BTC Up/Down prediction-market DQN project.

Phase 0 gives a clean per-episode dataset of resolved BTC 1-hour
Polymarket "Up or Down" markets, with intra-hour contract prices, spot-derived
features, time-to-resolution, and terminal labels, plus a chronological split.
"""

from . import config  # noqa: F401
