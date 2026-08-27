"""Training and inference for the student."""

from .sft import Example, HParams, train
from .infer import Student

__all__ = ["Example", "HParams", "train", "Student"]
