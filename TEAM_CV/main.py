# -*- coding: utf-8 -*-
"""KMU Smart Parking Monitor — Entrypoint

Usage:
  python main.py                 # host=127.0.0.1 port=8000 reload=on
  HOST=0.0.0.0 PORT=8000 python main.py
  python main.py --host 0.0.0.0 --port 8000 --no-reload

Environment variables (override CLI):
  HOST, PORT, RELOAD (1/0/true/false), LOG_LEVEL (info/debug), APP (default "parking_app.api:app")
"""
import os
import argparse
import uvicorn

# Default small CPU thread caps (optional; comment these to compare performance)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

def str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "on")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "info"))
    parser.add_argument("--app", default=os.getenv("APP", "parking_app.api:app"))
    args = parser.parse_args()

    reload_flag = not (args.no_reload or str2bool(os.getenv("RELOAD", "true")) is False)

    uvicorn.run(
        args.app,
        host=args.host,
        port=args.port,
        reload=reload_flag,
        log_level=args.log_level,
    )

if __name__ == "__main__":
    main()
