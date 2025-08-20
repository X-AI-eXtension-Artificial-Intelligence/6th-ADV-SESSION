# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict, Any
import numpy as np

class YOLODetector:
    def __init__(self, cfg: Dict[str,Any]):
        self.ok = False
        self.conf = cfg["conf"]; self.iou = cfg["iou"]; self.imgsz = cfg["imgsz"]; self.device = cfg["device"]
        self.empty_names = set(cfg.get("empty_labels") or [])
        self.occ_names   = set(cfg.get("occupied_labels") or [])
        try:
            from ultralytics import YOLO  # type: ignore
            self.model = YOLO(cfg["weights"])
            self.names = getattr(self.model, "names", {}) or getattr(getattr(self.model, "model", None), "names", {})
            self.ok = True
            print("[YOLO] loaded:", cfg["weights"])
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self.model.predict(source=[dummy], imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                   device=self.device, verbose=False)
        except Exception as e:
            print("[WARN] YOLO load fail:", e)
            self.ok = False

    def infer_detections(self, frame_bgr: np.ndarray) -> List[Tuple[float,float,str]]:
        if not self.ok:
            return []
        outs: List[Tuple[float,float,str]] = []
        try:
            res = self.model.predict(source=[frame_bgr], imgsz=self.imgsz,
                                     conf=self.conf, iou=self.iou,
                                     device=self.device, verbose=False)
            if not res: return outs
            r = res[0]
            boxes = getattr(r, 'boxes', None)
            if boxes is None: return outs
            xyxy = boxes.xyxy.cpu().numpy()
            cls  = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else np.full((len(xyxy),), -1)
            for i, (x1,y1,x2,y2) in enumerate(xyxy):
                name = self.names.get(int(cls[i]), str(cls[i]))
                cx = float((x1+x2)/2.0); cy = float((y1+y2)/2.0)
                outs.append((cx, cy, name))
        except Exception as e:
            print("[YOLO] infer err:", e)
        return outs
