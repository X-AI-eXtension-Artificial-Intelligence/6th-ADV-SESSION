# -*- coding: utf-8 -*-
import threading, time
from typing import List, Dict, Tuple, Any

class Slot:
    def __init__(self, slot_id: int, rect_norm: Tuple[float,float,float,float]):
        self.id = slot_id
        self.rect = rect_norm
        self.state = "empty"

    def contains_norm(self, cxn: float, cyn: float) -> bool:
        x1,y1,x2,y2 = self.rect
        return (x1 <= cxn <= x2) and (y1 <= cyn <= y2)

class Region:
    def __init__(self, region_id: str, slot_rects: Dict[int, Tuple[float,float,float,float]]):
        self.id = region_id
        self.slots: Dict[int, Slot] = {sid: Slot(sid, rect) for sid, rect in slot_rects.items()}
        self.lock = threading.Lock(); self.last_update = time.time()

    def update_with_detections(self, dets, frame_w:int, frame_h:int, empty_names:set, occ_names:set):
        observed: Dict[int, str] = {}
        for (cx, cy, name) in dets:
            cxn, cyn = cx/frame_w, cy/frame_h
            for sid, slot in self.slots.items():
                if slot.contains_norm(cxn, cyn):
                    if name in occ_names:
                        observed[sid] = "occupied"
                    elif (name in empty_names) and (observed.get(sid) != "occupied"):
                        observed[sid] = "empty"
                    break

        with self.lock:
            if not observed:
                for slot in self.slots.values():
                    slot.state = "empty"
            else:
                for sid, slot in self.slots.items():
                    new_state = observed.get(sid, slot.state)
                    slot.state = new_state
            self.last_update = time.time()

    def snapshot(self) -> Dict[str,Any]:
        with self.lock:
            return {
                "id": self.id,
                "slots": [{"id": sid, "rect_norm": s.rect, "state": s.state}
                          for sid,s in sorted(self.slots.items())],
                "last_update": self.last_update
            }
