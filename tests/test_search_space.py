# ABOUTME: Tests for SearchSpace — construction, sampling bounds, and YAML/dict round-trips.

import pytest

from llm_hyperopt import SearchSpace


def test_add_and_len():
    space = SearchSpace()
    space.add_float("lr", 1e-5, 1e-1, log=True)
    space.add_int("bs", 8, 256)
    space.add_categorical("optimizer", ["adam", "sgd", "adamw"])
    assert len(space) == 3


def test_sample_respects_bounds():
    space = SearchSpace()
    space.add_float("lr", 1e-5, 1e-1, log=True)
    space.add_int("bs", 8, 256)
    space.add_categorical("optimizer", ["adam", "sgd"])
    for _ in range(200):
        cfg = space.sample()
        assert 1e-5 <= cfg["lr"] <= 1e-1
        assert 8 <= cfg["bs"] <= 256
        assert cfg["optimizer"] in ("adam", "sgd")


def test_yaml_round_trip():
    space = SearchSpace()
    space.add_float("lr", 1e-5, 1e-1, log=True)
    space.add_categorical("optimizer", ["adam", "sgd", "adamw"])

    restored = SearchSpace.from_yaml(space.to_yaml())
    assert len(restored) == 2
    assert restored.parameters["lr"].to_dict() == {
        "type": "float",
        "min": 1e-5,
        "max": 1e-1,
        "log": True,
    }
    assert restored.parameters["optimizer"].to_dict() == {
        "type": "categorical",
        "values": ["adam", "sgd", "adamw"],
    }


def test_from_dict_rejects_unknown_type():
    with pytest.raises(ValueError):
        SearchSpace.from_dict({"parameters": {"x": {"type": "quantum"}}})
