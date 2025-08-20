# -*- coding: utf-8 -*-
import io, time, threading
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw
import cv2

from .config import YOLO_CFG, FLOORPLAN_PATH, FLOORPLAN_YOLO, VIDEO_YOLO, REGION_MAP, ZONE_MAP
from .detector import YOLODetector
from .models import Region

def _yolo_to_rect_norm(cx, cy, w, h):
    return (cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0)

def _build_rect_dict_norm(yolo_list):
    return {sid: _yolo_to_rect_norm(cx, cy, w, h) for sid, cx, cy, w, h in yolo_list}

class AppState:
    def __init__(self):
        self.frame_skip = YOLO_CFG["FRAME_SKIP"]
        # Build rect dicts for floorplan/video
        self.floorplan_rects = _build_rect_dict_norm(FLOORPLAN_YOLO)
        self.video_rects     = _build_rect_dict_norm(VIDEO_YOLO)

        # Regions: each uses VIDEO rects
        self.regions = {
            rid: Region(rid, {sid: self.video_rects[sid] for sid in sids})
            for rid, sids in REGION_MAP.items()
        }
        self.detector = YOLODetector(YOLO_CFG)

        self.base = self._load_floorplan()
        self.threads: List[threading.Thread] = []; self.stops: List[threading.Event] = []
        self.session_id: Optional[str] = None
        self._frame_jpeg: Dict[str, bytes] = {rid: b"" for rid in self.regions.keys()}
        self._frame_lock: Dict[str, threading.Lock] = {rid: threading.Lock() for rid in self.regions.keys()}

    def _load_floorplan(self):
        if not FLOORPLAN_PATH.exists():
            print("[WARN] floorplan missing -> blank canvas")
            return Image.new("RGB", (1500, 450), (245,245,245))
        return Image.open(FLOORPLAN_PATH).convert("RGB")

    def reset(self):
        self.stop()
        for reg in self.regions.values():
            for s in reg.slots.values():
                s.state="empty"
        self.session_id = str(time.time())

    def start(self, video_paths: List[str]):
        self.reset()
        order = ["cam0","cam1","cam2","cam3"]
        self.stops = []; self.threads = []
        for i, vp in enumerate(video_paths):
            if i >= 4: break
            rid = order[i]
            ev = threading.Event(); self.stops.append(ev)
            th = threading.Thread(target=self._worker, args=(rid, vp, ev), daemon=True)
            th.start(); self.threads.append(th)
        print("[APP] session start:", self.session_id)

    def stop(self):
        for e in self.stops: e.set()
        for t in self.threads: t.join(timeout=0.5)
        self.stops = []; self.threads = []

    def set_frame(self, cam_id: str, bgr: np.ndarray, det_centers: List[Tuple[float,float]]):
        frame = bgr.copy()
        h, w = frame.shape[:2]
        color_green = (0,255,0); color_blue = (255,0,0)

        # draw slots (VIDEO rects) in green
        for s in self.regions[cam_id].snapshot()["slots"]:
            x1n,y1n,x2n,y2n = s["rect_norm"]
            x1,y1,x2,y2 = int(x1n*w), int(y1n*h), int(x2n*w), int(y2n*h)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color_green, 2)

        # draw centers (blue)
        for (cx, cy) in det_centers:
            cv2.circle(frame, (int(cx), int(cy)), 3, color_blue, -1)

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ok:
            with self._frame_lock[cam_id]:
                self._frame_jpeg[cam_id] = buf.tobytes()

    def get_frame(self, cam_id: str) -> bytes:
        with self._frame_lock[cam_id]:
            return self._frame_jpeg.get(cam_id, b"")

    def _worker(self, region_id: str, video_path: str, stop: threading.Event):
        reg = self.regions[region_id]
        det = self.detector
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[ERR] open video fail:", video_path); return
        frame_id = 0

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

            # small buffer drain
            for _ in range(2):
                _ok, _f = cap.read()
                if not _ok: break
                frame = _f

            h, w = frame.shape[:2]
            target = YOLO_CFG["imgsz"]
            if max(h, w) > target * 1.2:
                if w >= h:
                    frame = cv2.resize(frame, (target, int(target * h / w)))
                else:
                    frame = cv2.resize(frame, (int(target * w / h), target))

            if frame_id % self.frame_skip == 0:
                dets = det.infer_detections(frame) if det.ok else []
                reg.update_with_detections(dets, frame.shape[1], frame.shape[0], det.empty_names, det.occ_names)
                centers_only = [(cx,cy) for (cx,cy,_) in dets]
                self.set_frame(region_id, frame, centers_only)
            frame_id += 1
        cap.release()

    def render(self):
        base = self._load_floorplan().copy().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        W,H = base.size
        green = (0,220,0); a=int(255*0.45)

        # aggregate states
        slot_state: Dict[int, str] = {}
        for reg in self.regions.values():
            snap = reg.snapshot()
            for s in snap["slots"]:
                slot_state[s["id"]] = s["state"]

        # Use floorplan rects
        for (sid, cx, cy, w, h) in FLOORPLAN_YOLO:
            if slot_state.get(sid, "empty") == "empty":
                x1n, y1n, x2n, y2n = (cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0)
                x1,y1,x2,y2 = int(x1n*W), int(y1n*H), int(x2n*W), int(y2n*H)
                draw.rectangle([x1,y1,x2,y2],
                               fill=(green[0],green[1],green[2],a),
                               outline=(green[0],green[1],green[2],255), width=2)
        return Image.alpha_composite(base, overlay).convert("RGB")

    def _summary(self) -> Dict[str, Any]:
        slot_state: Dict[int, str] = {}
        for reg in self.regions.values():
            snap = reg.snapshot()
            for s in snap["slots"]:
                slot_state[s["id"]] = s["state"]

        def zone_stat(slot_ids):
            total = len(slot_ids)
            empty = sum(1 for sid in slot_ids if slot_state.get(sid, "empty") == "empty")
            occ   = total - empty
            occ_pct = int(round(100 * occ / total)) if total else 0
            return {"total": total, "empty": empty, "occupied": occ, "occupancy": occ_pct}

        b226 = zone_stat(ZONE_MAP["B2 26"])
        b227 = zone_stat(ZONE_MAP["B2 27"])
        overall = {
            "total": b226["total"] + b227["total"],
            "empty": b226["empty"] + b227["empty"],
            "occupied": b226["occupied"] + b227["occupied"],
            "occupancy": int(round(
                100 * (b226["occupied"] + b227["occupied"]) / (b226["total"] + b227["total"])
            ))
        }
        return {"B2 26": b226, "B2 27": b227, "overall": overall}

    def state(self):
        return {
            "session": self.session_id,
            "regions": [self.regions[r].snapshot() for r in ["cam0","cam1","cam2","cam3"]],
            "summary": self._summary(),
        }

STATE = AppState()
