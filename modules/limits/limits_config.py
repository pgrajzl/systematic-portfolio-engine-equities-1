"""
limits_config.py

Pure configuration -- no functions, just definitions. These constants
are the hard constraints that optimization/ will take, alongside
current weights and factor exposures, to solve for a final portfolio
that respects all of them simultaneously.

This file only DEFINES the limits. It does not enforce them --
enforcement (including intelligent redistribution when a position
needs to be trimmed) is optimization/'s job.
"""

# --- Position size limits ---
# Maximum absolute weight (long or short) any single stock can hold,
# as a fraction of total portfolio gross exposure.
MAX_POSITION_WEIGHT = 0.05  # no single stock > 5% of the book

# --- Exposure limits ---
# Target total gross exposure (sum of absolute value of every position).
# 1.0 = fully invested, 100% of capital deployed (matches the
# convention used in construction/'s gross_exposure parameter).
TARGET_GROSS_EXPOSURE = 1.0

# Maximum allowed deviation from zero net exposure (dollar-neutrality).
# A small nonzero band is allowed rather than forcing an exact 0,
# since exact neutrality isn't always achievable simultaneously with
# every other constraint.
MAX_NET_EXPOSURE = 0.02  # net long/short imbalance capped at 2% of gross

# --- Factor exposure limits ---
# Maximum allowed portfolio-level beta exposure (post-neutralization,
# this should ideally be near 0, but a small tolerance band is set
# here rather than requiring an exact 0).
MAX_BETA_EXPOSURE = 0.10

# Maximum allowed net exposure to any single sector.
MAX_SECTOR_EXPOSURE = 0.05  # no sector > 5% net long or short

# --- Turnover control ---
# Minimum change in a position's target weight (as a fraction of
# gross exposure) required before it's actually traded. Below this
# threshold, the position is left unchanged from the prior rebalance,
# to avoid churning on noisy, immaterial signal changes.
MIN_TRADE_THRESHOLD = 0.001  # 0.1% of gross exposure

# --- Universe / diversification limits (optional, worth having available) ---
# Minimum number of positions on each side (long/short) required for
# a rebalance to be considered valid -- guards against a degenerate
# portfolio concentrated in just a handful of names.
MIN_POSITIONS_PER_SIDE = 10