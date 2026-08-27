"""Answering the sealed test set with the student, on vLLM.

Both model-backed arms come through here: A2 carries a long prompt and no
adapter, A6 carries an adapter and no prompt, A7 may carry both.  One
generation path means a difference between arms is a difference in the
artifact rather than in how it was run.

**Prefix caching is a fairness requirement, not an optimisation.**  A2's
whole cost story is that its prompt is re-paid on every one of 10^4 queries.
Real deployments prefill a shared prefix once and reuse its KV cache, so
measuring A2 without that would inflate its cost several times over and
manufacture the crossover this experiment is supposed to discover -- the plan
calls the disabled version a straw man.  vLLM does this natively.

**One engine per process, adapters swapped per request.**  A6 builds a
student several times in a single run -- once for each dev evaluation inside
the agent loop, once more for the sealed test set -- and standing an engine
up and tearing it down each time went wrong in three separate ways before
this shape was settled on:

  fork could not inherit an initialised CUDA context, and A6 trains before it
  generates, so the GPU is always live by then;

  spawn avoided that but re-imports the entry module in the child, so every
  driver script had to guard `__main__` or re-run itself;

  in-process fixed both, and then tearing the engine down between students
  left vLLM's distributed state half-initialised -- "Process group is not
  initialized in the world group map" -- on the next one.

Reusing the engine removes the whole class of problem, and it is what vLLM's
`LoRARequest` is for.  It is also much faster: reloading 1.7B parameters
between dev evaluations was minutes of GPU time buying nothing.

Sampling is greedy.  Test-time decoding is not one of the variables under
study, and letting it vary would put noise straight into the crossover.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# The engine runs in this process. See the module docstring for why; the
# spawn setting remains a fallback for anyone who overrides the first.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

#: One engine per (base model, shape). Built on first use, never torn down --
#: a process runs one arm, and the engine dies with it.
_ENGINES: dict[tuple, object] = {}
_ADAPTER_IDS: dict[str, int] = {}


@dataclass
class Generation:
    answers: list[str]
    prefix_tokens: int
    generated_tokens: int
    seconds: float
    prefix_caching: bool


def _engine(base_model: str, max_model_len: int, gpu_memory_utilization: float,
            max_lora_rank: int, dtype: str):
    key = (base_model, max_model_len, gpu_memory_utilization, max_lora_rank, dtype)
    if key not in _ENGINES:
        import gc

        import torch
        from vllm import LLM

        # A6 has just finished training; the optimiser state and model are
        # dead but still resident, and vLLM sizes its KV cache from what it
        # sees free. Without this the student gets a fraction of the memory
        # it should and the arm looks slower than it is.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _ENGINES[key] = LLM(
            model=base_model, dtype=dtype,
            enable_prefix_caching=True,      # the fairness requirement above
            # Always on: an engine built without it cannot serve an adapter
            # later, and A2 and A6 may share a process in a grid run.
            enable_lora=True, max_lora_rank=max_lora_rank,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_log_stats=True,
        )
    return _ENGINES[key]


class Student:
    """The small model, optionally with a context prefix and/or an adapter."""

    def __init__(self, base_model: str, *, adapter_path: str | None = None,
                 context: str | None = None, max_new_tokens: int = 24,
                 max_model_len: int = 8192, gpu_memory_utilization: float = 0.85,
                 max_lora_rank: int = 64, dtype: str = "bfloat16"):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(base_model)
        self.context = context
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.llm = _engine(base_model, max_model_len, gpu_memory_utilization,
                           max_lora_rank, dtype)

        self._lora = None
        if adapter_path:
            from vllm.lora.request import LoRARequest
            if adapter_path not in _ADAPTER_IDS:
                _ADAPTER_IDS[adapter_path] = len(_ADAPTER_IDS) + 1
            self._lora = LoRARequest(f"student{_ADAPTER_IDS[adapter_path]}",
                                     _ADAPTER_IDS[adapter_path], adapter_path)

    @property
    def prefix_tokens(self) -> int:
        if not self.context:
            return 0
        return len(self.tok(self.context, add_special_tokens=False)["input_ids"])

    def close(self) -> None:
        """Drop this student's handle. The engine stays up for the next one.

        Tearing the engine down here is what broke A6: vLLM's distributed
        state does not survive being re-initialised in the same process.
        """
        self.llm = None

    def answer(self, exprs: list[str], *, ledger=None, prompt_of=None,
               **_ignored) -> Generation:
        """Greedy-decode an answer per expression.

        Batching is vLLM's business: the whole list goes in at once and the
        scheduler packs it. The context prefix is identical across every
        prompt, so it is prefilled once and reused.
        """
        from vllm import SamplingParams

        prompt_of = prompt_of or (lambda e: f"{e} =")
        head = self.context or ""
        prompts = [f"{head}{prompt_of(e)}" for e in exprs]

        params = SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens)
        timer = ledger.gpu_timer("student_inference") if ledger is not None else None
        if timer is not None:
            timer.__enter__()
        try:
            outs = self.llm.generate(
                prompts, params,
                **({"lora_request": self._lora} if self._lora else {}))
        finally:
            if timer is not None:
                timer.__exit__(None, None, None)

        answers, generated = [], 0
        for o in outs:
            generated += len(o.outputs[0].token_ids)
            answers.append(_clean(o.outputs[0].text))

        seconds = 0.0
        if ledger is not None:
            # gpu_timer charges kind "gpu_second" with the label in meta, so
            # filtering on kind alone would pick up training time too.
            recs = [r for r in ledger.records
                    if r.kind == "gpu_second"
                    and r.meta.get("label") == "student_inference"]
            seconds = recs[-1].n if recs else 0.0
        return Generation(answers, self.prefix_tokens, generated, seconds, True)


_ANSWER = re.compile(r"(\[[^\]]*\]|v(?:_[0-9a-z]+)+|v\d+)")


def _clean(text: str) -> str:
    """First well-formed value or list in the model's output.

    A trained student emits a bare answer; the untrained one in the A8 arm
    will happily add prose. Taking the first value-shaped thing is the same
    leniency for every arm, which is the part that matters.
    """
    m = _ANSWER.search(text)
    return m.group(1) if m else text.strip()
