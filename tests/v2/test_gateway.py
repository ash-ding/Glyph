"""Tests for the metering model gateway (src/glyph/v2/gateway.py).

Async-tooling note: this env has neither pytest-asyncio nor pytest-aiohttp installed
(checked via `python -c "import pytest_asyncio"` / `import aiohttp.pytest_plugin` and via
importlib.metadata entry_points -- no pytest11 plugin registers `aiohttp_client` or an
`asyncio` marker here). Per the task brief, we do NOT add new test deps; instead the gateway's
request-handling core is a plain async function (`Gateway.handle_request`) with no aiohttp.web
server dependency, and we drive it directly with `anyio.run` (already installed, dep-free) from
plain synchronous test functions. This keeps the three required behaviours -- forward+meter,
disallowed model -> 403 not billed, retry-then-success bills once -- fully exercised without any
async pytest fixtures/markers. A small extra test checks that `make_app()` still builds a real
aiohttp Application for production use.
"""

import json

import anyio

from glyph.v2.gateway import Gateway
from glyph.v2.ledger import Ledger

PIN = "claude-opus-4-8"


def _resp(usage):
    """A minimal Vertex-style (non-streamed) response body carrying `usage`."""
    return json.dumps({"type": "message", "usage": usage}).encode()


def test_allowed_model_forwarded_and_metered():
    ledger = Ledger()

    async def fwd(method, url, headers, body):
        assert "/v1/" in url  # the CLI-dropped /v1 prefix must be re-inserted
        return (
            200,
            {},
            _resp(
                {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 5,
                    "cache_creation_input_tokens": 3,
                }
            ),
        )

    gw = Gateway(ledger, pinned_model=PIN, forward_fn=fwd)

    async def run():
        path = f"/projects/p/locations/global/publishers/anthropic/models/{PIN}:streamRawPredict"
        status, _headers, _body = await gw.handle_request("POST", path, {}, b"{}")
        assert status == 200

    anyio.run(run)

    k = ledger.summary()["usd_by_kind"]
    assert k["input"] > 0 and k["output"] > 0 and k["cache_read"] > 0 and k["cache_write"] > 0
    assert gw.records[-1]["allowed"] is True and gw.records[-1]["input_tokens"] == 10


def test_disallowed_model_403_not_billed():
    ledger = Ledger()

    async def fwd(*a, **k):
        raise AssertionError("must not forward")

    gw = Gateway(ledger, pinned_model=PIN, forward_fn=fwd)

    async def run():
        path = "/projects/p/locations/global/publishers/anthropic/models/claude-haiku-4-5:streamRawPredict"
        status, _headers, _body = await gw.handle_request("POST", path, {}, b"{}")
        assert status == 403

    anyio.run(run)

    assert ledger.spent_usd == 0.0
    assert gw.records[-1]["allowed"] is False


def test_retry_bills_once():
    ledger = Ledger()
    calls = {"n": 0}

    async def fwd(method, url, headers, body):
        calls["n"] += 1
        if calls["n"] == 1:
            return 500, {}, b"err"
        return (
            200,
            {},
            _resp(
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            ),
        )

    gw = Gateway(ledger, pinned_model=PIN, forward_fn=fwd)

    async def run():
        path = f"/projects/p/locations/global/publishers/anthropic/models/{PIN}:streamRawPredict"
        status, _headers, _body = await gw.handle_request("POST", path, {}, b"{}")
        assert status == 200

    anyio.run(run)

    assert calls["n"] == 2
    # billed exactly once: 1 input token + 1 output token, no double-billing on the retry
    assert abs(ledger.spent_usd - (1 * 5e-6 + 1 * 25e-6)) < 1e-12


def test_make_app_builds_a_real_aiohttp_application():
    """Sanity check for the production wiring path -- no async fixtures required to build it."""
    from aiohttp import web

    ledger = Ledger()

    async def fwd(method, url, headers, body):
        return (
            200,
            {},
            _resp(
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            ),
        )

    gw = Gateway(ledger, pinned_model=PIN, forward_fn=fwd)
    app = gw.make_app()
    assert isinstance(app, web.Application)


def test_forward_exception_becomes_retryable_503_not_a_hang():
    """A forward that raises (upstream timeout / conn reset) must be caught and
    returned as a retryable 503 -- never propagated to a silent aiohttp 500 that
    stalls the CLI (the train-arm E2E hang). It is retried MAX_ATTEMPTS times and
    bills nothing."""
    from glyph.v2.gateway import MAX_ATTEMPTS

    ledger = Ledger()
    calls = {"n": 0}

    async def fwd(method, url, headers, body):
        calls["n"] += 1
        raise TimeoutError("upstream stalled")

    gw = Gateway(ledger, pinned_model=PIN, forward_fn=fwd)

    async def run():
        path = f"/projects/p/locations/global/publishers/anthropic/models/{PIN}:streamRawPredict"
        status, _h, body = await gw.handle_request("POST", path, {}, b"{}")
        assert status == 503
        assert b"upstream forward failed" in body

    anyio.run(run)

    assert calls["n"] == MAX_ATTEMPTS  # retried, not abandoned after one attempt
    assert ledger.summary()["usd_by_kind"].get("input", 0) == 0  # billed nothing
    assert gw.records[-1]["allowed"] is True and gw.records[-1]["status"] == 503


def test_forward_programming_bug_propagates_not_masked_as_503():
    """A non-network exception from forward_fn (a real bug, e.g. ValueError)
    must propagate out of handle_request, not be swallowed into a retryable 503
    that hides it and makes the client retry three times over a broken path."""
    ledger = Ledger()

    async def fwd(method, url, headers, body):
        raise ValueError("bug in forward path")

    gw = Gateway(ledger, pinned_model=PIN, forward_fn=fwd)

    async def run():
        path = f"/projects/p/locations/global/publishers/anthropic/models/{PIN}:streamRawPredict"
        await gw.handle_request("POST", path, {}, b"{}")

    raised = False
    try:
        anyio.run(run)
    except ValueError:
        raised = True
    assert raised, "a non-network forward bug must propagate, not become a 503"
    assert ledger.summary()["usd_by_kind"].get("input", 0) == 0

def test_parse_thinking_non_streamed():
    import json as _json
    from glyph.v2.gateway import _parse_thinking

    body = _json.dumps(
        {
            "type": "message",
            "content": [
                {"type": "thinking", "thinking": "let me reason about this problem step by step"},
                {"type": "tool_use", "id": "toolu_abc123", "name": "run_tests", "input": {}},
                {"type": "text", "text": "Here is the answer you asked for."},
            ],
        }
    )
    rec = _parse_thinking(body)
    assert rec["thinking"] == ["let me reason about this problem step by step"]
    assert rec["text_preview"].startswith("Here is the answer")
    assert rec["tool_use_ids"] == ["toolu_abc123"]
    assert rec["redacted"] == 0


def test_parse_thinking_streamed_sse():
    from glyph.v2.gateway import _parse_thinking

    lines = [
        'data: {"type":"message_start","message":{"id":"msg_1"}}',
        '',
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}',
        '',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"first part, "}}',
        '',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"second part"}}',
        '',
        'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_xyz789","name":"run_tests"}}',
        '',
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{}"}}',
        '',
        'data: {"type":"content_block_stop","index":1}',
        '',
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
        '',
    ]
    body = "\n".join(lines)
    rec = _parse_thinking(body)
    assert rec["thinking"] == ["first part, second part"]
    assert rec["tool_use_ids"] == ["toolu_xyz789"]
    assert rec["redacted"] == 0


def test_parse_thinking_redacted_non_streamed():
    import json as _json
    from glyph.v2.gateway import _parse_thinking

    body = _json.dumps({"type": "message", "content": [{"type": "redacted_thinking", "data": "encrypted-blob"}]})
    rec = _parse_thinking(body)
    assert rec["thinking"] == []
    assert rec["redacted"] >= 1


def test_parse_thinking_redacted_streamed():
    from glyph.v2.gateway import _parse_thinking

    body = '\n'.join(
        [
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking"}}',
            '',
            'data: {"type":"content_block_stop","index":0}',
            '',
        ]
    )
    rec = _parse_thinking(body)
    assert rec["thinking"] == []
    assert rec["redacted"] >= 1


def test_parse_thinking_malformed_body_does_not_raise():
    from glyph.v2.gateway import _parse_thinking

    rec = _parse_thinking("not json at all {{{")
    assert rec["thinking"] == []
    assert rec["redacted"] == 0
    assert rec["tool_use_ids"] == []
    assert rec["text_preview"] == ""


def test_gateway_writes_thinking_file_when_present(tmp_path):
    import json as _json
    import anyio

    from glyph.v2.gateway import Gateway
    from glyph.v2.ledger import Ledger

    thinking_path = tmp_path / "thinking.jsonl"

    async def fwd(method, url, headers, body):
        return (
            200,
            {},
            _json.dumps(
                {
                    "type": "message",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "content": [
                        {"type": "thinking", "thinking": "pondering deeply"},
                        {"type": "text", "text": "done"},
                    ],
                }
            ).encode(),
        )

    gw = Gateway(
        Ledger(),
        pinned_model="claude-opus-4-8",
        forward_fn=fwd,
        thinking_log_path=str(thinking_path),
    )

    async def run():
        path = "/projects/p/locations/global/publishers/anthropic/models/claude-opus-4-8:streamRawPredict"
        status, _h, _b = await gw.handle_request("POST", path, {}, b"{}")
        assert status == 200

    anyio.run(run)

    assert thinking_path.exists()
    lines = thinking_path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = _json.loads(lines[0])
    assert rec["thinking"] == ["pondering deeply"]


def test_gateway_writes_nothing_when_no_thinking(tmp_path):
    import json as _json
    import anyio

    from glyph.v2.gateway import Gateway
    from glyph.v2.ledger import Ledger

    thinking_path = tmp_path / "thinking.jsonl"

    async def fwd(method, url, headers, body):
        return (
            200,
            {},
            _json.dumps(
                {
                    "type": "message",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "content": [{"type": "text", "text": "no thinking here"}],
                }
            ).encode(),
        )

    gw = Gateway(
        Ledger(),
        pinned_model="claude-opus-4-8",
        forward_fn=fwd,
        thinking_log_path=str(thinking_path),
    )

    async def run():
        path = "/projects/p/locations/global/publishers/anthropic/models/claude-opus-4-8:streamRawPredict"
        status, _h, _b = await gw.handle_request("POST", path, {}, b"{}")
        assert status == 200

    anyio.run(run)

    assert not thinking_path.exists()
