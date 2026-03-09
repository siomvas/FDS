#!/usr/bin/env python3
"""Quick test of track_epoch and get_round_statistics methods."""

import logging
from fl_training.nvflare_refactor.common.gradient_tracker import ClientGradientTracker
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# Create dummy model
model = nn.Linear(10, 5)

# Create tracker
tracker = ClientGradientTracker()
tracker.store_initial_weights(model)

# Simulate gradient computation
for param in model.parameters():
    param.grad = torch.randn_like(param) * 0.01

# Track epoch
tracker.track_epoch(model, 0)
logger.info("track_epoch() executed successfully")

# Get round statistics
stats = tracker.get_round_statistics()
logger.info("get_round_statistics() returned %d keys", len(stats))
logger.info("Keys: %s", sorted(stats.keys()))

# Verify epoch_history cleared
stats2 = tracker.get_round_statistics()
logger.info("Second call returned %d keys (expected 0)", len(stats2))

assert len(stats) > 0, "Stats should not be empty"
assert len(stats2) == 0, "Second call should return empty dict"

logger.info("All verifications passed.")
