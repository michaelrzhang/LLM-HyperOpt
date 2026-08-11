# ABOUTME: Typed representation of a hyperparameter search space (float/int/categorical),
# ABOUTME: with YAML and dict round-tripping and random sampling for baselines/fallback.

import math
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field


class Parameter(BaseModel, ABC):
    """Base class for optimization parameters."""

    name: str

    @abstractmethod
    def sample(self) -> Any:
        """Sample a random value from this parameter's domain."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter to dictionary representation."""


class FloatParameter(Parameter):
    """Continuous floating-point parameter."""

    min_value: float = Field(alias="min")
    max_value: float = Field(alias="max")
    log: bool = False

    def sample(self) -> float:
        if self.log:
            return math.exp(
                random.uniform(math.log(self.min_value), math.log(self.max_value))
            )
        return random.uniform(self.min_value, self.max_value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "float",
            "min": self.min_value,
            "max": self.max_value,
            "log": self.log,
        }


class IntParameter(Parameter):
    """Discrete integer parameter."""

    min_value: int = Field(alias="min")
    max_value: int = Field(alias="max")
    log: bool = False

    def sample(self) -> int:
        if self.log:
            return int(
                math.exp(
                    random.uniform(
                        math.log(max(1, self.min_value)), math.log(self.max_value)
                    )
                )
            )
        return random.randint(self.min_value, self.max_value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "int",
            "min": self.min_value,
            "max": self.max_value,
            "log": self.log,
        }


class CategoricalParameter(Parameter):
    """Categorical parameter with discrete choices."""

    values: List[Any]

    def sample(self) -> Any:
        return random.choice(self.values)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "categorical", "values": self.values}


class SearchSpace:
    """Defines the hyperparameter search space for optimization."""

    def __init__(self) -> None:
        self.parameters: Dict[str, Parameter] = {}

    def add_float(
        self, name: str, min_value: float, max_value: float, log: bool = False
    ) -> None:
        """Add a continuous floating-point parameter."""
        self.parameters[name] = FloatParameter(
            name=name, min=min_value, max=max_value, log=log
        )

    def add_int(
        self, name: str, min_value: int, max_value: int, log: bool = False
    ) -> None:
        """Add a discrete integer parameter."""
        self.parameters[name] = IntParameter(
            name=name, min=min_value, max=max_value, log=log
        )

    def add_categorical(self, name: str, values: List[Any]) -> None:
        """Add a categorical parameter with a fixed set of choices."""
        self.parameters[name] = CategoricalParameter(name=name, values=values)

    def sample(self) -> Dict[str, Any]:
        """Sample a random configuration. Used for random-search baselines."""
        return {name: param.sample() for name, param in self.parameters.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameters": {
                name: param.to_dict() for name, param in self.parameters.items()
            }
        }

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SearchSpace":
        """Build a search space from a ``{"parameters": {...}}`` dict."""
        space = cls()
        for name, cfg in config["parameters"].items():
            param_type = cfg["type"]
            if param_type == "float":
                space.add_float(name, cfg["min"], cfg["max"], cfg.get("log", False))
            elif param_type == "int":
                space.add_int(name, cfg["min"], cfg["max"], cfg.get("log", False))
            elif param_type == "categorical":
                space.add_categorical(name, cfg["values"])
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        return space

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "SearchSpace":
        """Build a search space from a YAML definition."""
        return cls.from_dict(yaml.safe_load(yaml_content))

    def __len__(self) -> int:
        return len(self.parameters)

    def __repr__(self) -> str:
        return f"SearchSpace({len(self.parameters)} parameters)"
