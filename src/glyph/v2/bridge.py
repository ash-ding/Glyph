"""In-sandbox bridge: runs inside the bwrap sandbox as the CLI's launcher.

Starts a background thread listening on 127.0.0.1:8787. Each TCP connection
accepted there is proxied, bidirectionally, to the unix domain socket
/run/gw.sock (the gateway socket bind-mounted read-only into the sandbox by
wrapper.sh). The bridge then os.execv's the real CLI with the argv given
after "--", per the task-15 interface.

The CLI is exec'd unconditionally: even if it never opens a connection to
127.0.0.1:8787, and even if /run/gw.sock turns out not to be connectable
(dialing gw.sock only happens lazily, per accepted client, inside
_handle_client) the exec still happens. A stub CLI that makes no network
calls at all (as tests/v2/test_isolation.py's stub does) must still run to
completion.

KNOWN LIMITATION: os.execv() replaces this whole process image, which kills
the listener thread along with it. The listening socket's fd itself can
survive the exec (Python sockets are non-inheritable/CLOEXEC by default, so
in fact it does NOT survive -- accept() and the port both disappear at
exec). Any connection to 127.0.0.1:8787 that the CLI opens *after* exec
therefore cannot be served by this process. This differs from the spike
(/tmp/glyph_v2_spikes/bridge.py), which stays resident as the CLI's actual
parent (asyncio.create_subprocess_exec + wait) so the proxy keeps running
for the CLI's whole lifetime. Task 15's interface explicitly calls for
os.execv, and the isolation regression test in tests/v2/test_isolation.py
exercises only the "exec always happens" guarantee with a stub CLI that
never dials 8787 -- so this limitation does not affect that test. It DOES
mean a real end-to-end run (an actual CLI making Vertex calls through this
bridge) needs the resident/subprocess model instead; flagged for follow-up
rather than silently resolved here, since the brief's given interface is
unambiguous about using execv.

Usage: python3 bridge.py -- <cli> <cli-args...>

stdlib only.
"""
from __future__ import annotations

import os
import socket
import sys
import threading

GATEWAY_SOCK = "/run/gw.sock"
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


def start_listener(host: str = LISTEN_HOST, port: int = LISTEN_PORT) -> threading.Thread:
    """Bind and listen immediately (in this thread, so callers can rely on
    the port being open once this returns), then hand off accept() to a
    background daemon thread. Returns that thread."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(128)

    thread = threading.Thread(target=_serve_forever, args=(server,), daemon=True)
    thread.start()
    return thread


def main(argv: list[str]) -> None:
    if "--" not in argv:
        sys.stderr.write("usage: bridge.py -- <cli> <cli-args...>\n")
        sys.exit(2)
    cli_argv = argv[argv.index("--") + 1 :]
    if not cli_argv:
        sys.stderr.write("usage: bridge.py -- <cli> <cli-args...>\n")
        sys.exit(2)

    start_listener()

    # Per the task-15 interface: exec the CLI unconditionally, replacing
    # this process. See the KNOWN LIMITATION note in the module docstring --
    # the listener thread does not survive this call.
    os.execv(cli_argv[0], cli_argv)


if __name__ == "__main__":
    main(sys.argv[1:])
