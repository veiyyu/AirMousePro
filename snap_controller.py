
# snap_controller.py
import math

class SnapController:
    def __init__(self, enter_px=22, exit_px=54):
        self.ENTER_PX = enter_px
        self.EXIT_PX = exit_px
        self.current_target = None
        self.is_snapped = False
        self.last_dist = 999.0

    def update(self, raw_x, raw_y, target_data):
        target = target_data or self.current_target
        dist = self._dist_to_bounds(raw_x, raw_y, target["bounds"]) if target else 999.0
        self.last_dist = dist

        if not self.is_snapped:
            if dist <= self.ENTER_PX:
                self.is_snapped = True
                self.current_target = target
        else:
            if dist > self.EXIT_PX:
                self.is_snapped = False
                self.current_target = None
            elif target_data and target_data != self.current_target:
                new_dist = self._dist_to_bounds(raw_x, raw_y, target_data["bounds"])
                if new_dist < dist * 0.7:
                    self.current_target = target_data

        if self.is_snapped and self.current_target:
            bx, by, bw, bh = self.current_target["bounds"]
            # Precision Upgrade: Snap to the nearest point in the rectangle rather than the fixed center
            tx = min(max(raw_x, bx), bx + bw)
            ty = min(max(raw_y, by), by + bh)
            
            alpha = max(0, min(1, 1 - (dist / self.EXIT_PX)))
            alpha = alpha ** 2.0 
            dx = (1 - alpha) * raw_x + alpha * tx
            dy = (1 - alpha) * raw_y + alpha * ty
            return True, (dx, dy)
        return False, (raw_x, raw_y)

    def _dist_to_bounds(self, px, py, bounds):
        x, y, w, h = bounds
        dx = max(x - px, 0, px - (x + w))
        dy = max(y - py, 0, py - (y + h))
        return math.sqrt(dx*dx + dy*dy)
