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
calls the disabled version a straw man.  vLLM does this natively via
`enable_prefix_caching`, which is why generation runs here rather than
through a hand-rolled cache: a cache written by us would be one more thing
that has to be proven equivalent to the uncached path before any A2 number
can be trusted.

Sampling is greedy (`temperature=0`).  Test-time decoding is not one of the
variables under study, and letting it vary would put noise straight into the
crossover estimate.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# The engine runs in this process, not a subprocess.  Two footguns disappear
# with it, and both had already gone off:
#
#   fork cannot inherit an initialised CUDA context, and A6 trains before it
#   generates, so the GPU is always live by the time the student is built --
#   "Cannot re-initialize CUDA in forked subprocess".
#
#   spawn avoids that but re-imports the entry module in the child, so every
#   driver script would have to guard `if __name__ == "__main__"` or the
#   whole run re-executes. A library that only works when the caller
#   remembers a guard is a library that will break on someone's ad-hoc
#   script, and it did.
#
# In-process costs the engine's isolation, which buys nothing here: this is
# offline batch generation of one model, not a served endpoint.  The spawn
# setting stays as a fallback for anyone who overrides the first variable.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@dataclass
class Generation:
    answers: list[str]
    prefix_tokens: int
    generated_tokens: int
    seconds: float
    prefix_caching: bool


class Student:
    """The small model, optionally with a context prefix and/or an adapter."""

    def __init__(self, base_model: str, *, adapter_path: str | None = None,
                 context: str | None = None, max_new_tokens: int = 24,
                 max_model_len: int = 8192, gpu_memory_utilization: float = 0.85,
                 max_lora_rank: int = 64, dtype: str = "bfloat16"):
        import gc

        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        # A6 has just finished training; its optimiser state and model are
        # dead but still resident, and vLLM sizes its KV cache from what it
        # sees free. Without this the student gets a fraction of the memory
        # it should and the arm looks slower than it is.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.tok = AutoTokenizer.from_pretrained(base_model)
        self.context = context
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens

        self.llm = LLM(
            model=base_model,
            dtype=dtype,
            enable_prefix_caching=True,      # the fairness requirement above
            enable_lora=bool(adapter_path),
            max_lora_rank=max_lora_rank if adapter_path else 16,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_log_stats=True,
        )
        self._lora = None
        if adapter_path:
            from vllm.lora.request import LoRARequest
            self._lora = LoRARequest("student", 1, adapter_path)

    @property
    def prefix_tokens(self) -> int:
        if not self.context:
            return 0
        return len(self.tok(self.context, add_special_tokens=False)["input_ids"])

    def close(self) -> None:
        """Release the engine so the next Student can have the memory.

        Running in-process means the engine's NCCL group belongs to us and
        outlives the object unless it is torn down; leaving it up leaks the
        allocation the next student needs.
        """
        import contextlib
        import gc

        import torch
        self.llm = None
        with contextlib.suppress(Exception):
            from vllm.distributed import destroy_model_parallel
            destroy_model_parallel()
        with contextlib.suppress(Exception):
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def answer(self, exprs: list[str], *, ledger=None, prompt_of=None,
               **_ignored) -> Generation:
        """Greedy-decode an answer per expression.

        Batching is vLLM's business: the whole list goes in at once and the
        scheduler packs it. The shared context prefix is identical across
        every prompt, so prefix caching prefills it once and every subsequent
        query reuses those blocks.
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
            outs = self.llm.generate(prompts, params,
                                     **({"lora_request": self._lora} if self._lora else {}))
        finally:
            if timer is not None:
                timer.__exit__(None, None, None)

        # vLLM returns results keyed to the input order it was given, but the
        # scheduler reorders internally; request_id is the safe join key.
        by_id = {o.request_id: o for o in outs}
        ordered = [by_id.get(str(i), outs[i]) for i in range(len(prompts))] \
            if len(by_id) == len(prompts) and all(str(i) in by_id for i in range(len(prompts))) \
            else list(outs)

        answers, generated = [], 0
        for o in ordered:
            text = o.outputs[0].text
            generated += len(o.outputs[0].token_ids)
            answers.append(_clean(text))

        seconds = 0.0
        if ledger is not None:
            # gpu_timer charges kind "gpu_second" and puts the label in meta,
            # so filtering on kind alone would pick up training time too.
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
