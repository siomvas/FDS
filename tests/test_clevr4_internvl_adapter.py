from PIL import Image
import pytest

from partitioning.vlm.internvl_adapter import (
    dynamic_preprocess,
    find_closest_aspect_ratio,
    resolve_device_map,
)


def test_dynamic_preprocess_matches_native_clevr_aspect_ratio():
    image = Image.new("RGB", (640, 480))
    tiles = dynamic_preprocess(image, max_tiles=12, image_size=32)

    assert len(tiles) == 13
    assert all(tile.size == (32, 32) for tile in tiles)


def test_dynamic_preprocess_rejects_invalid_tile_limits():
    image = Image.new("RGB", (640, 480))
    with pytest.raises(ValueError, match="tile limits"):
        dynamic_preprocess(image, min_tiles=2, max_tiles=1)


def test_aspect_ratio_tie_prefers_more_tiles_for_sufficient_area():
    ratio = find_closest_aspect_ratio(
        1.0,
        [(1, 1), (2, 2)],
        width=1024,
        height=1024,
        image_size=32,
    )
    assert ratio == (2, 2)


def test_device_map_streams_single_gpu_weights_directly():
    assert resolve_device_map("cuda:0", 1) == {"": "cuda:0"}
    assert resolve_device_map("cuda:0", 2) == "balanced"
    assert resolve_device_map("auto", 0) == "auto"
    assert resolve_device_map("cpu", 0) is None
