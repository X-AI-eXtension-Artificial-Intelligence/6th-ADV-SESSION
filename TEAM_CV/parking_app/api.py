# -*- coding: utf-8 -*-
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .config import ROOT, TEMPLATES_DIR, STATIC_DIR, UPLOAD_DIR
from .state import STATE
import os, time, io

app = FastAPI(title="KMU Smart Parking Monitor")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Mount static dir
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    p = TEMPLATES_DIR / "index.html"
    return HTMLResponse(p.read_text(encoding="utf-8"))

@app.post("/upload")
async def upload(cam0: UploadFile = File(...), cam1: UploadFile = File(...),
                 cam2: UploadFile = File(...), cam3: UploadFile = File(...)):
    files = [cam0, cam1, cam2, cam3]
    paths = []
    ts = int(time.time())
    for i, uf in enumerate(files):
        ext = os.path.splitext(uf.filename or f"cam{i}.mp4")[1] or ".mp4"
        p = UPLOAD_DIR / f"session_{ts}_{i}{ext}"
        with open(p, "wb") as f:
            f.write(await uf.read())
        paths.append(str(p))
    STATE.start(paths)
    return JSONResponse({"ok": True, "paths": paths, "session": STATE.session_id})

@app.post("/stop")
def stop():
    STATE.stop()
    return {"ok": True}

@app.get("/state")
def state():
    return JSONResponse(STATE.state())

@app.get("/render")
def render():
    img = STATE.render()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

def mjpeg_generator(cam_id: str):
    boundary = b"--frame"
    while True:
        jpg = STATE.get_frame(cam_id)
        if not jpg:
            import time as _t; _t.sleep(0.05); continue
        yield boundary + b"\r\n" + b"Content-Type: image/jpeg\r\n" + \
              b"Content-Length: " + str(len(jpg)).encode() + \
              b"\r\n\r\n" + jpg + b"\r\n"
        import time as _t; _t.sleep(0.08)

@app.get("/stream/{cam_id}")
def stream(cam_id: str):
    if cam_id not in STATE.regions:
        raise HTTPException(status_code=404, detail="invalid cam id")
    return StreamingResponse(
        mjpeg_generator(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
