"""In-sandbox bridge: runs inside the bwrap sandbox as the CLI's launcher.

Starts a background thread listening on 127.0.0.1:8787. Each TCP connection
accepted there is proxied, bidirectionally, to the unix domain socket at
GATEWAY_SOCK (by default /run/gw.sock, the gateway socket bind-mounted
read-only into the sandbox by wrapper.sh; overridable via the
GLYPH_GW_SOCK env var for testing -- see tests/v2/test_bridge.py).

RESIDENT MODEL (fix for task-15 round 1): the bridge stays alive as the
CLI's actual parent process. It starts the listener, then runs the real CLI
as a *subprocess* (subprocess.Popen) and waits for it, exiting with the
CLI's own return code. The listener thread therefore survives for the
CLI's entire lifetime, so 127.0.0.1:8787 keeps proxying to GATEWAY_SOCK for
as long as the CLI is running.

(An earlier version of this file replaced this process outright with the
CLI via a process-image-replacing exec call. That was wrong: replacing the
process image that way kills the listener thread along with it, so any
connection the CLI opened to 127.0.0.1:8787 *after* that point could never
be served -- breaking the real end-to-end path entirely. This matches the
S1 Stage D spike (/tmp/glyph_v2_spikes/bridge.py), which also stays
resident via asyncio.create_subprocess_exec + wait rather than replacing
itself with the CLI.)

The CLI is still run unconditionally, even if it never opens a connection
to 127.0.0.1:8787 and even if GATEWAY_SOCK turns out not to be connectable
(dialing it only happens lazily, per accepted client, inside
_handle_client): a stub CLI that makes no network calls at all (as
tests/v2/test_isolation.py's stub does) still runs to completion as this
process's child, and the bridge exits with its return code.

Usage: python3 bridge.py -- <cli> <cli-args...>

stdlib only.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading

# Overridable for tests (tests/v2/test_bridge.py points this at a temp AF_UNIX
# echo server instead of the real gateway socket). Production always uses the
# default -- wrapper.sh never sets GLYPH_GW_SOCK, so GATEWAY_SOCK is
# /run/gw.sock inside the sandbox, matching sandbox.py's --ro-bind target.
GATEWAY_SOCK = os.environ.get("GLYPH_GW_SOCK", "/run/gw.sock")
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8787
_CHUNK = 65536


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    """Copy bytes from src to dst until EOF or error, then half-close dst."""
    try:
        while True:
            data = src.recv(_CHUNK)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _handle_client(client: socket.socket) -> None:
    try:
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        upstream.connect(GATEWAY_SOCK)
    except OSError as exc:
        sys.stderr.write("bridge: upstream connect to %s failed: %s\n" % (GATEWAY_SOCK, exc))
        try:
            client.close()
        except OSError:
            pass
        return

    t1 = threading.Thread(target=_pipe, args=(client, upstream), daemon=True)
    t2 = threading.Thread(target=_pipe, args=(upstream, client), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    for s in (client, upstream):
        try:
            s.close()
        except OSError:
            pass


def _serve_forever(server: socket.socket) -> None:
    while True:
        try:
            client, _addr = server.accept()
        except OSError:
            return
        threading.Thread(target=_handle_client, args=(client,), daemon=True).start()


def start_listener(host: str = LISTEN_HOST, port: int = LISTEN_PORT) -> tuple[threading.Thread, int]:
    """Bind and listen immediately (in this thread, so callers can rely on
    the port being open once this returns), then hand off accept() to a
    background daemon thread. Returns (thread, bound_port) -- pass port=0
    for an ephemeral port (as tests do) and read the actual port back from
    the second element."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(128)
    bound_port = server.getsockname()[1]

    thread = threading.Thread(target=_serve_forever, args=(server,), daemon=True)
    thread.start()
    return thread, bound_port


def main(argv: list[str]) -> None:
    if "--" not in argv:
        sys.stderr.write("usage: bridge.py -- <cli> <cli-args...>\n")
        sys.exit(2)
    cli_argv = argv[argv.index("--") + 1 :]
    if not cli_argv:
        sys.stderr.write("usage: bridge.py -- <cli> <cli-args...>\n")
        sys.exit(2)

    start_listener()

    # Resident model: run the CLI as a child and stay alive as its parent so
    # the listener thread (and thus the 8787 -> GATEWAY_SOCK proxy) survives
    # for the CLI's whole lifetime. Exit with the CLI's own return code.
    proc = subprocess.Popen(cli_argv)
    rc = proc.wait()
    sys.exit(rc)


if __name__ == "__main__":
    main(sys.argv[1:])
