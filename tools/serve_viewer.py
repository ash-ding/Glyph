#!/usr/bin/env python3
"""Collect runs and serve the viewer in one command — built for remote use.

    python tools/serve_viewer.py --port 8000 /tmp/glyph_runs

Collects every `run.json` under the given roots into `glyph-viewer/runs/`, then serves
the repo over HTTP on 127.0.0.1:<port>. Bound to localhost only, so it is
reachable just through an SSH tunnel:

    # on your laptop:
    ssh -N -L 8000:localhost:8000 lumen1
    # then open http://localhost:8000/glyph-viewer/

No local repo, no scp. New runs show up on a browser refresh: each request for
the run index re-collects first.
"""
import argparse
import functools
import http.server
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_runs import collect  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "glyph-viewer" / "runs"


def make_handler(roots):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(REPO_ROOT), **k)

        def do_GET(self):
            # re-collect so a refresh always reflects the latest runs
            if self.path.rstrip("/").endswith("/glyph-viewer/runs/index.json"):
                try:
                    collect(roots, OUT_DIR)
                except Exception as e:  # never take the server down over a bad run
                    sys.stderr.write("re-collect failed: %r\n" % e)
            return super().do_GET()

        def log_message(self, fmt, *args):
            pass  # quiet

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=["runs"],
                    help="directories to scan for run.json (default: ./runs)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1 — localhost only, use an SSH tunnel)")
    args = ap.parse_args(argv)

    idx = collect(args.roots, OUT_DIR)
    print("collected %d run(s) from %s" % (len(idx), ", ".join(args.roots)))
    print("serving %s on http://%s:%d" % (REPO_ROOT, args.host, args.port))
    print("open      http://localhost:%d/glyph-viewer/" % args.port)
    print("tunnel    ssh -N -L %d:localhost:%d <this-host>   # from your laptop"
          % (args.port, args.port))
    httpd = http.server.ThreadingHTTPServer((args.host, args.port), make_handler(args.roots))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
