
# overlay_cursor.py
from AppKit import (
    NSWindow, NSView, NSColor, NSBezierPath, NSScreen,
    NSBackingStoreBuffered, NSWindowStyleMaskBorderless,
    NSScreenSaverWindowLevel
)

try:
    from Quartz import CGDisplayHideCursor, CGDisplayShowCursor, CGMainDisplayID
except Exception:
    CGDisplayHideCursor = None
    CGDisplayShowCursor = None
    CGMainDisplayID = None

class CursorOverlayView(NSView):
    def drawRect_(self, rect):
        if not getattr(self, "state", None): return
        pos = self.state.get("pos", (0, 0))
        snapped = self.state.get("snapped", False)
        target_bounds = self.state.get("target_bounds")
        screen_frame = self.window().screen().frame()
        screen_h = screen_frame.size.height

        if snapped and target_bounds:
            bx, by, bw, bh = target_bounds
            padding = 6
            appkit_y = screen_h - (by + bh)
            draw_rect = ((bx - padding, appkit_y - padding), (bw + padding * 2, bh + padding * 2))
            radius = min((bh + padding * 2) / 2.0, 14.0)
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(draw_rect, radius, radius)
            NSColor.colorWithDeviceRed_green_blue_alpha_(0.35, 0.60, 1.00, 0.18).set()
            path.fill()
            NSColor.colorWithDeviceRed_green_blue_alpha_(0.45, 0.70, 1.00, 0.65).set()
            path.setLineWidth_(2.5)
            path.stroke()
        else:
            cx, cy = pos
            radius = 8
            appkit_y = screen_h - cy
            draw_rect = ((cx - radius, appkit_y - radius), (radius * 2, radius * 2))
            path = NSBezierPath.bezierPathWithOvalInRect_(draw_rect)
            NSColor.colorWithDeviceRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.85).set()
            path.setLineWidth_(2.0)
            path.stroke()
            dot_rect = ((cx - 1.6, appkit_y - 1.6), (3.2, 3.2))
            dot = NSBezierPath.bezierPathWithOvalInRect_(dot_rect)
            NSColor.colorWithDeviceRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.9).set()
            dot.fill()

class CursorOverlayManager:
    def __init__(self):
        screen_frame = NSScreen.mainScreen().frame()
        self.win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            screen_frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
        )
        self.win.setOpaque_(False)
        self.win.setBackgroundColor_(NSColor.clearColor())
        self.win.setLevel_(NSScreenSaverWindowLevel + 2)
        self.win.setIgnoresMouseEvents_(True)
        self.win.setHasShadow_(False)
        self.view = CursorOverlayView.alloc().initWithFrame_(screen_frame)
        self.win.setContentView_(self.view)
        self.win.orderFrontRegardless()
        self._hide_cursor()

    def _hide_cursor(self):
        try:
            if CGDisplayHideCursor and CGMainDisplayID:
                CGDisplayHideCursor(CGMainDisplayID())
        except Exception: pass

    def _show_cursor(self):
        try:
            if CGDisplayShowCursor and CGMainDisplayID:
                CGDisplayShowCursor(CGMainDisplayID())
        except Exception: pass

    def update(self, snapped, pos, target_bounds):
        self.view.state = {"snapped": snapped, "pos": pos, "target_bounds": target_bounds}
        self.view.setNeedsDisplay_(True)

    def cleanup(self):
        self._show_cursor()
        self.win.close()
