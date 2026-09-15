"""Proxy round-trip test for the in-sandbox bridge (glyph.v2.bridge).

No bwrap involved -- this exercises bridge.start_listener() directly, in
process, to prove the resident listener thread actually proxies TCP traffic
on 127.0.0.1:<port> through to a unix domain socket (standing in for the
real gateway socket, /run/gw.sock) and back. This is the regression test
for the round-1 bug: an earlier version of bridge.py replaced its own
process image with the CLI, which killed this exact listener thread and
silently broke the proxy for any connection made after that point. This
test would have caught that: it fails if the listener does not survive to
serve a connection.
"""
import os
import socket
import threading

import pytest


def _run_unix_echo_server(sock_path: str, ready: threading.Event) -> None:
    """Accept exactly one AF_UNIX connection and echo back whatever it
    sends, until the peer closes. Stands in for the real gateway socket."""
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    ready.set()
    conn, _ = srv.accept()
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            conn.sendall(data)
    finally:
        conn.close()
        srv.close()


def test_bridge_proxies_tcp_to_unix_socket_and_back(tmp_path, monkeypatch):
    gw = tmp_path / "gw.sock"

    # Point the bridge at our temp unix echo server instead of the real
    # gateway socket. GATEWAY_SOCK is read from this env var at import time,
    # so set it before importing (or reloading) the module.
    monkeypatch.setenv("GLYPH_GW_SOCK", str(gw))
    import importlib

    from glyph.v2 import bridge
    importlib.reload(bridge)  # pick up GLYPH_GW_SOCK even if already imported

    assert bridge.GATEWAY_SOCK == str(gw)

    ready = threading.Event()
    server_thread = threading.Thread(target=_run_unix_echo_server, args=(str(gw), ready), daemon=True)
    server_thread.start()
    assert ready.wait(timeout=5), "unix echo server never bound"

    listener_thread, port = bridge.start_listener(port=0)
    assert listener_thread.is_alive()
    assert port != 0

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(5)
    try:
        client.connect(("127.0.0.1", port))
        client.sendall(b"ping")
        data = client.recv(4096)
        assert data == b"ping"

        # A second round trip on the same connection, to show the proxy
        # keeps working beyond a single message (the listener is resident,
        # not one-shot).
        client.sendall(b"pong-pong")
        data2 = client.recv(4096)
        assert data2 == b"pong-pong"
    finally:
        client.close()

    # The listener thread is a daemon thread serving accept() in a loop; it
    # is still alive after our client disconnected (ready for more
    # connections), which is exactly the "survives the whole CLI lifetime"
    # property the round-1 fix restores.
    assert listener_thread.is_alive()
