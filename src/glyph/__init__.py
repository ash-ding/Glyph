"""Glyph -- a hidden-semantics DSL execution benchmark."""
from .data.config import GlyphConfig, PRESETS
from .data.instance import GlyphInstance, generate

__all__ = ["GlyphConfig", "PRESETS", "GlyphInstance", "generate"]
__version__ = "0.1.0"
