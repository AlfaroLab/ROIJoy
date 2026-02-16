import numpy as np
import pytest


def test_match_identical_images_returns_same_vertices():
    from roijoy.matching import match_roi

    img = np.random.RandomState(42).randint(0, 255, (200, 200, 3), dtype=np.uint8)
    vertices = [(50, 50), (100, 50), (100, 100), (50, 100)]
    result = match_roi(img, img, vertices)

    assert result["success"] is True
    assert len(result["vertices"]) == 4
    for orig, matched in zip(vertices, result["vertices"]):
        assert abs(orig[0] - matched[0]) < 5
        assert abs(orig[1] - matched[1]) < 5


def test_match_featureless_images_falls_back():
    from roijoy.matching import match_roi

    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    vertices = [(50, 50), (100, 50), (100, 100), (50, 100)]

    result = match_roi(blank, blank, vertices)
    assert result["success"] is True
    assert result["method"] == "copy"


def test_copy_roi_returns_exact_vertices():
    from roijoy.matching import copy_roi

    vertices = [(50.5, 60.7), (100.2, 50.3)]
    result = copy_roi(vertices)
    assert result["vertices"] == vertices
    assert result["method"] == "copy"
