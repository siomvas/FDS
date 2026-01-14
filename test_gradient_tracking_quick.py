#!/usr/bin/env python3
"""Quick test of track_epoch and get_round_statistics methods."""

from fl_training.nvflare_refactor.common.gradient_tracker import ClientGradientTracker
import torch
import torch.nn as nn

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
print("✓ track_epoch() executed successfully")

# Get round statistics
stats = tracker.get_round_statistics()
print(f"✓ get_round_statistics() returned {len(stats)} keys")
print(f"Keys: {sorted(stats.keys())}")

# Verify epoch_history cleared
stats2 = tracker.get_round_statistics()
print(f"✓ Second call returned {len(stats2)} keys (should be 0)")

assert len(stats) > 0, "Stats should not be empty"
assert len(stats2) == 0, "Second call should return empty dict"

print("\n✓ All verifications passed!")
