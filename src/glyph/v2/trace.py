"""Run trace adapter: thin wrapper over glyph.trace.TraceWriter."""

from pathlib import Path
from glyph.trace import TraceWriter


class RunTrace:
    """Adapter providing per-run JSONL trace with convenience helpers."""
    
    def __init__(self, run_dir):
        """Open run_dir/trace.jsonl via TraceWriter (creating run_dir if needed)."""
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.run_dir / "trace.jsonl"
        self._writer = TraceWriter(trace_path)
    
    def emit(self, kind: str, **fields) -> None:
        """Emit event to trace (delegates to TraceWriter)."""
        self._writer.emit(kind, **fields)
    
    def phase_event(self, phase: str, reason: str, **fields) -> None:
        """Emit a phase event with phase and reason."""
        self.emit("phase", phase=phase, reason=reason, **fields)
    
    def close(self) -> None:
        """Close the trace writer."""
        self._writer.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.close()
        return False
