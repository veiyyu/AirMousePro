
# snap_targeting.py
import math
from ApplicationServices import (
    AXUIElementCreateSystemWide,
    AXUIElementCopyElementAtPosition,
    AXUIElementCopyAttributeValue,
    AXUIElementPerformAction,
)

AX_ROLE = "AXRole"
AX_FRAME = "AXFrame"
AX_PARENT = "AXParent"
AX_ACTION_NAMES = "AXActionNames"

class SnapTargeting:
    ACTIONABLE_ROLES = {
        "AXButton", "AXLink", "AXTextField", "AXCheckBox", "AXRadioButton",
        "AXPopUpButton", "AXMenuItem", "AXTab", "AXComboBox"
    }

    def __init__(self, scan_radii=(0, 10, 16, 22), parent_depth=6):
        self.system_wide = AXUIElementCreateSystemWide()
        self.scan_radii = list(scan_radii)
        self.parent_depth = parent_depth

    def get_element_at(self, x, y):
        err, el = AXUIElementCopyElementAtPosition(self.system_wide, float(x), float(y), None)
        return el if err == 0 else None

    def find_actionable_target(self, x, y):
        x, y = float(x), float(y)
        for r in self.scan_radii:
            pts = [(x, y)] if r == 0 else self._circle_points(x, y, r, 8)
            for px, py in pts:
                el = self.get_element_at(px, py)
                if not el: continue
                actionable = self._walk_to_actionable(el)
                if not actionable: continue
                bounds = self._get_bounds(actionable)
                if bounds and self._is_valid_size(bounds):
                    return {"element": actionable, "bounds": bounds}
        return None

    def perform_click(self, target_data):
        if not target_data: return False
        el = target_data.get("element")
        try:
            AXUIElementPerformAction(el, "AXPress")
            return True
        except Exception: return False

    def _get_attr(self, el, attr_name):
        try:
            err, val = AXUIElementCopyAttributeValue(el, attr_name, None)
            return val if err == 0 else None
        except Exception: return None

    def _walk_to_actionable(self, el):
        cur = el
        for _ in range(self.parent_depth + 1):
            if self._is_actionable(cur): return cur
            parent = self._get_attr(cur, AX_PARENT)
            if not parent: break
            cur = parent
        return None

    def _is_actionable(self, el):
        role = self._get_attr(el, AX_ROLE)
        if role in self.ACTIONABLE_ROLES: return True
        actions = self._get_attr(el, AX_ACTION_NAMES)
        return actions and "AXPress" in actions

    def _get_bounds(self, el):
        frame = self._get_attr(el, AX_FRAME)
        if not frame: return None
        try:
            return (float(frame.origin.x), float(frame.origin.y),
                    float(frame.size.width), float(frame.size.height))
        except Exception: return None

    def _is_valid_size(self, bounds):
        _, _, w, h = bounds
        return 2 < w < 1400 and 2 < h < 1000

    def _circle_points(self, cx, cy, r, n):
        pts = []
        for i in range(n):
            a = (2 * math.pi * i) / n
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return pts
