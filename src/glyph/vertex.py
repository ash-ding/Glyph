"""Teacher access through Google Vertex AI.

Every arm that has a frontier agent reaches it here.  This project talks to
Vertex rather than the first-party Anthropic API because that is where its
credentials live -- and because **Opus 5 is not available on this Vertex
project**.  Opus 4.8 is the main teacher; Opus 4.6 is the cross-replication
teacher for E7; the two 4.5-generation models exist only for the E9
capability ladder and are not interchangeable with either Opus.

Auth is GCP Application Default Credentials, not an Anthropic API key:

    gcloud auth application-default login

Three things below are experiment-critical rather than stylistic.

**`thinking` is passed explicitly, always.**  On Opus 4.8 and 4.6, omitting
the parameter means the model does not think at all, and nothing errors.
Forgetting it would run every arm against a quietly weaker teacher than the
plan specifies -- the kind of failure that shows up as a result, not as a
traceback.

**`effort` is a controlled variable.**  It is a budget lever, so it is fixed
at one level for every arm and swept only in E7.  Letting an arm choose it
would re-open a dimension of the action space this design deliberately
closed.

**Request construction branches by model in exactly one place.**  The
4.5-generation models take neither `effort` nor adaptive thinking; they are
on the older `budget_tokens` interface.  Scattering that difference across
call sites is how the ladder ends up not comparable to itself.
"""

from __future__ import annotations

import os
from typing import Any

from anthropic import AnthropicVertex

# --- Vertex endpoint ------------------------------------------------------
# Defaults are lumen-3's; the env vars are what the shell already exports, so
# a different host overrides without touching this file.
PROJECT_ID = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "itpc-ca-273afd9a9f")
REGION = os.environ.get("CLOUD_ML_REGION", "global")

# --- teachers -------------------------------------------------------------
TEACHER = "claude-opus-4-8"          # main teacher, every arm
CROSS_TEACHER = "claude-opus-4-6"    # E7: same tier, different generation

# E9 only.  Sonnet and Haiku are NOT same-tier substitutes for the cross
# teacher: swapping one in would confound "a different model" with "a weaker
# model", which is the whole point E9 measures on purpose and E7 must not.
LADDER: tuple[str, ...] = (TEACHER, CROSS_TEACHER, "claude-sonnet-4-5", "claude-haiku-4-5")

#: Fixed across all arms.  Swept only in E7.
EFFORT = "high"

#: Models on the adaptive-thinking + effort interface.
_ADAPTIVE = frozenset({TEACHER, CROSS_TEACHER})

#: Thinking budget for the 4.5-generation ladder models, which have no
#: `effort`.  Must be >= 1024 and strictly less than `max_tokens`.
_LADDER_THINKING_TOKENS = 4096


def request_kwargs(model: str, max_tokens: int = 16000,
                   effort: str | None = None) -> dict[str, Any]:
    """The per-model half of a request, so no call site has to know the shape.

    Returns `model`, `max_tokens`, `thinking`, and -- on the Opus models --
    `output_config`.  Merge with `messages` / `system` / `tools` and hand
    straight to `messages.create` or `messages.stream`.
    """
    if model in _ADAPTIVE:
        return {
            "model": model,
            "max_tokens": max_tokens,
            # Explicit, never omitted: omitting it means no thinking at all.
            "thinking": {"type": "adaptive"},
            # `effort` lives inside output_config, not at the top level.
            # Arms must leave this alone -- it is a controlled variable, fixed
            # across every arm and swept only in E7. The override exists for
            # the self-checks, which are not arms: #4 at `high` spends its
            # whole output budget thinking and never reaches an answer, and a
            # teacher that thinks itself out of a reply has demonstrated a
            # token budget, not hiddenness.
            "output_config": {"effort": effort or EFFORT},
        }

    # 4.5 generation: fixed thinking budget, and `effort` is rejected.
    budget = min(_LADDER_THINKING_TOKENS, max_tokens - 1)
    if budget < 1024:
        raise ValueError(
            f"max_tokens={max_tokens} leaves no room for a thinking budget on "
            f"{model}: it needs >= 1024 and strictly less than max_tokens."
        )
    return {
        "model": model,
        "max_tokens": max_tokens,
        "thinking": {"type": "enabled", "budget_tokens": budget},
    }


def client(project_id: str = PROJECT_ID, region: str = REGION) -> AnthropicVertex:
    return AnthropicVertex(project_id=project_id, region=region)


def chat(
    prompt: str,
    *,
    system: str | None = None,
    model: str = TEACHER,
    max_tokens: int = 16000,
    vertex: AnthropicVertex | None = None,
) -> str:
    """One turn, streamed.  Convenience for probes -- the orchestrator owns
    its own loop so it can meter every call through the ledger."""
    vertex = vertex or client()
    kwargs = request_kwargs(model, max_tokens)
    if system:
        kwargs["system"] = system
    # Streamed because these models allow very large max_tokens, and a
    # non-streaming request that big trips the SDK's HTTP timeout.
    with vertex.messages.stream(messages=[{"role": "user", "content": prompt}], **kwargs) as stream:
        message = stream.get_final_message()
    if message.stop_reason == "refusal":
        raise RuntimeError(f"refused: {message.stop_details}")
    return "".join(b.text for b in message.content if b.type == "text")


if __name__ == "__main__":
    import sys

    print(f"project {PROJECT_ID}  region {REGION}\n")
    vertex = client()
    failed = 0
    for model in LADDER:
        tag = {TEACHER: "teacher", CROSS_TEACHER: "cross-teacher (E7)"}.get(model, "ladder (E9)")
        try:
            kwargs = request_kwargs(model, max_tokens=2048)
            with vertex.messages.stream(
                messages=[{"role": "user", "content": "Reply with the single word: ready"}],
                **kwargs,
            ) as stream:
                msg = stream.get_final_message()
            text = "".join(b.text for b in msg.content if b.type == "text").strip()
            u = msg.usage
            print(f"  {model:<22} {tag:<20} OK   {text!r}  "
                  f"in={u.input_tokens} out={u.output_tokens}")
        except Exception as exc:  # noqa: BLE001 - a probe reports, it does not handle
            failed += 1
            first = str(exc).strip().splitlines()[0][:150]
            print(f"  {model:<22} {tag:<20} FAIL {type(exc).__name__}: {first}")

    print()
    if failed == len(LADDER):
        print("nothing reachable -- check ADC (`gcloud auth application-default login`).")
        sys.exit(1)
    print(f"{len(LADDER) - failed}/{len(LADDER)} reachable.")
    sys.exit(0)
