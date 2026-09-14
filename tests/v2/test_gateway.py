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
