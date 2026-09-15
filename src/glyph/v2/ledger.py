"""Ledger for tracking model costs, GPU/CPU seconds in protocol v2."""

# Cost rates (from v1 src/glyph/budget.py CostModel)
USD_PER_IN_TOKEN = 5.0e-6
USD_PER_OUT_TOKEN = 25.0e-6
CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25
USD_PER_H100_SECOND = 0.00111  # informational only


class Ledger:
    """Accumulates model cost (by token kind), GPU/CPU seconds, and safety checks."""

    def __init__(self):
        """Initialize ledger with zero tracking."""
        self.usd_by_kind = {
            "input": 0.0,
            "cache_read": 0.0,
            "cache_write": 0.0,
            "output": 0.0,
        }
        self.gpu_seconds = {}  # label -> accumulated seconds
        self.cpu_seconds = 0.0

    def add_model_usage(self, *, input: int, cache_read: int, cache_write: int, output: int) -> None:
        """Add model token usage and accumulate costs by kind."""
        self.usd_by_kind["input"] += input * USD_PER_IN_TOKEN
        self.usd_by_kind["cache_read"] += cache_read * USD_PER_IN_TOKEN * CACHE_READ_MULT
        self.usd_by_kind["cache_write"] += cache_write * USD_PER_IN_TOKEN * CACHE_WRITE_MULT
        self.usd_by_kind["output"] += output * USD_PER_OUT_TOKEN

    def add_gpu_seconds(self, label: str, n: float) -> None:
        """Add GPU seconds for a given label (accumulates if label exists)."""
        if label not in self.gpu_seconds:
            self.gpu_seconds[label] = 0.0
        self.gpu_seconds[label] += n

    def add_cpu_seconds(self, n: float) -> None:
        """Add CPU seconds."""
        self.cpu_seconds += n

    @property
    def spent_usd(self) -> float:
        """Return total USD spent on model tokens only (excludes GPU/CPU costs)."""
        return sum(self.usd_by_kind.values())

    def over_safety_line(self, limit_usd: float) -> bool:
        """Check if spent_usd exceeds the given limit."""
        return self.spent_usd > limit_usd

    def summary(self) -> dict:
        """Return summary of all tracked costs and times."""
        gpu_summary = dict(self.gpu_seconds)  # copy current gpu_seconds
        gpu_summary["_total"] = sum(self.gpu_seconds.values())
        
        return {
            "usd_by_kind": dict(self.usd_by_kind),
            "spent_usd": self.spent_usd,
            "gpu_seconds": gpu_summary,
            "cpu_seconds": self.cpu_seconds,
        }
