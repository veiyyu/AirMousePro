
import cv2
import threading
import sys
import math
import time

from snap_targeting import SnapTargeting
from snap_controller import SnapController
from overlay_cursor import CursorOverlayManager

try:
    import numpy as np
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except ImportError:
    print("FATAL: mediapipe missing.")
    sys.exit(1)

try:
    from Quartz import (
        CGEventCreateMouseEvent, CGEventPost, kCGHIDEventTap,
        kCGEventMouseMoved, kCGEventLeftMouseDown, kCGEventLeftMouseUp,
        kCGMouseButtonLeft, kCGEventFlagMaskCommand, kCGEventFlagMaskControl,
        kCGEventFlagMaskAlternate, kCGEventKeyDown, kCGEventKeyUp, CGEventMaskBit,
        CGEventTapCreate, kCGHeadInsertEventTap, kCGEventTapOptionDefault,
        CFMachPortCreateRunLoopSource, CFRunLoopAddSource, CFRunLoopGetCurrent,
        kCFRunLoopCommonModes, CFRunLoopRun, CGEventGetIntegerValueField,
        CGEventGetFlags, kCGKeyboardEventKeycode, kCGMouseEventClickState,
        CGEventSetIntegerValueField, kCGEventLeftMouseDragged,
        CGEventCreateScrollWheelEvent, kCGScrollEventUnitPixel,
        CGEventCreateKeyboardEvent, CGEventSetFlags
    )
    from AppKit import NSScreen
except ImportError:
    print("FATAL: pyobjc missing.")
    sys.exit(1)

# --- CONFIG & TUNING ---
DRAG_HOLD_S = 0.26           # Faster drag start
DRAG_RELEASE_CONFIRM_S = 0.10 # Hysteresis for drag release
SCROLL_MULT = 18.0
HOTKEY_KEYCODE = 35 # 'P'
HOTKEY_FLAGS = kCGEventFlagMaskCommand | kCGEventFlagMaskControl | kCGEventFlagMaskAlternate
TWO_FINGER_FREEZE_S = 0.22
MAX_STEP_PX = 95.0
ACTIVE_MIN = 0.12
ACTIVE_MAX = 0.88
SCROLL_DEADZONE_PX = 6

# --- PRECISION & ADAPTIVE SMOOTHING CONFIG ---
ALPHA_SLOW = 0.18    # Heavy smoothing for precision hovering
ALPHA_FAST = 0.55    # Snappy response for rapid movement
FAST_DIST_PX = 60.0  # Distance threshold to transition to fast alpha
DEADBAND_PX = 2.2    # Ignore micro-tremors below this pixel delta

# --- GESTURE THRESHOLDS (SENSITIVITY TWEAKS) ---
PINCH_ON  = 0.36
PINCH_OFF = 0.44
MIDDLE_PINCH_ON  = 0.40
MIDDLE_PINCH_OFF = 0.52
TWO_FINGER_ON  = 0.35
TWO_FINGER_OFF = 0.47
INDEX_CLEAR_MARGIN = 0.06

# --- FIST GESTURE CONFIG ---
FIST_HOLD_S = 1.0
FIST_COOLDOWN_S = 1.5
FIST_RATIO_ON = 0.55  # Folded tip distance / hand size

screen_frame = NSScreen.mainScreen().frame()
SCREEN_W, SCREEN_H = int(screen_frame.size.width), int(screen_frame.size.height)

class AirMouseState:
    def __init__(self):
        self.running = True
        self.paused = False
        self.x, self.y = SCREEN_W/2, SCREEN_H/2
        self.obs_x, self.obs_y = SCREEN_W/2, SCREEN_H/2
        self.active_x, self.active_y = self.x, self.y
        
        # Snap State
        self.snap_x, self.snap_y = SCREEN_W/2, SCREEN_H/2
        self.snapped = False
        self.target_data = None
        self.last_ax_query_t = 0
        
        # Gestures
        self.dragging = False
        self.drag_release_start = None
        self.scroll_mode = False
        self.two_finger_active = False
        self.pinch_active = False
        self.pinch_start_t = 0
        self.freeze_until = 0
        self.last_scroll_emit = 0
        self.index_pinched = False
        self.last_click_t = 0
        self.last_double_t = 0
        self.did_drag_this_pinch = False
        self.drag_anchor_x, self.drag_anchor_y = 0, 0
        self.scroll_start_t = 0
        
        # Fist / Minimize
        self.fist_start_t = None
        self.last_fist_action_t = 0.0

state = AirMouseState()
snap_targeting = SnapTargeting()
snap_controller = SnapController()
overlay = CursorOverlayManager()

def post_mouse(etype, x, y, clicks=1):
    ev = CGEventCreateMouseEvent(None, etype, (x, y), kCGMouseButtonLeft)
    CGEventSetIntegerValueField(ev, kCGMouseEventClickState, clicks)
    CGEventPost(kCGHIDEventTap, ev)

def post_scroll(dx, dy):
    ev = CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitPixel, 2, int(dy), int(dx))
    CGEventPost(kCGHIDEventTap, ev)

def double_click(x, y):
    post_mouse(kCGEventLeftMouseDown, x, y, clicks=1)
    post_mouse(kCGEventLeftMouseUp, x, y, clicks=1)
    time.sleep(0.015)
    post_mouse(kCGEventLeftMouseDown, x, y, clicks=2)
    post_mouse(kCGEventLeftMouseUp, x, y, clicks=2)

def send_cmd_m():
    KEY_M = 46
    ev_down = CGEventCreateKeyboardEvent(None, KEY_M, True)
    CGEventSetFlags(ev_down, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, ev_down)
    ev_up = CGEventCreateKeyboardEvent(None, KEY_M, False)
    CGEventSetFlags(ev_up, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, ev_up)

def is_closed_fist(lms, hand_size):
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17]
    ds = []
    for t, m in zip(tips, mcps):
        d = math.hypot(lms[t].x - lms[m].x, lms[t].y - lms[m].y) / hand_size
        ds.append(d)
    return (sum(ds) / len(ds)) < FIST_RATIO_ON

def action_pos():
    return float(state.active_x), float(state.active_y)

def cursor_thread():
    """120Hz Precision Dispatcher with Adaptive EMA and Deadband Filter"""
    while state.running:
        start_t = time.perf_counter()
        if not state.paused:
            # Calculate Delta for Adaptive Filtering
            dx = state.obs_x - state.x
            dy = state.obs_y - state.y
            dist_delta = math.hypot(dx, dy)

            if dist_delta < DEADBAND_PX:
                pass
            else:
                t_factor = min(1.0, dist_delta / FAST_DIST_PX)
                current_alpha = ALPHA_SLOW + (ALPHA_FAST - ALPHA_SLOW) * t_factor
                state.x += dx * current_alpha
                state.y += dy * current_alpha
            
            # Decide visual point
            current_active_x, current_active_y = state.x, state.y
            if state.snapped and not (state.dragging or state.scroll_mode or state.two_finger_active):
                current_active_x, current_active_y = state.snap_x, state.snap_y
            
            state.active_x, state.active_y = current_active_x, current_active_y

            if state.dragging:
                post_mouse(kCGEventLeftMouseDragged, state.active_x, state.active_y)
            else:
                post_mouse(kCGEventMouseMoved, state.active_x, state.active_y)
                
        time.sleep(max(0, (1/120.0) - (time.perf_counter() - start_t)))

threading.Thread(target=cursor_thread, daemon=True).start()

def hotkey_loop():
    def cb(proxy, type_, event, refcon):
        if type_ == kCGEventKeyDown:
            if CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode) == HOTKEY_KEYCODE:
                if (CGEventGetFlags(event) & HOTKEY_FLAGS) == HOTKEY_FLAGS:
                    state.paused = not state.paused
                    return None
        return event
    tap = CGEventTapCreate(kCGHIDEventTap, kCGHeadInsertEventTap, kCGEventTapOptionDefault, CGEventMaskBit(kCGEventKeyDown), cb, None)
    if tap:
        source = CFMachPortCreateRunLoopSource(None, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes)
        CFRunLoopRun()

threading.Thread(target=hotkey_loop, daemon=True).start()

cap = cv2.VideoCapture(1)
detector = mp_hands.Hands(
    model_complexity=0, 
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

try:
    while cap.isOpened() and state.running:
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)
        
        now = time.time()
        mode_str = "NO HAND"
        
        if res.multi_hand_landmarks:
            mode_str = "READY"
            lms = res.multi_hand_landmarks[0].landmark
            mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            
            hand_size = math.sqrt((lms[5].x - lms[17].x)**2 + (lms[5].y - lms[17].y)**2)
            hand_size = max(hand_size, 0.01)
            def dist_n(a, b): return math.sqrt((lms[a].x-lms[b].x)**2 + (lms[a].y-lms[b].y)**2) / hand_size
            
            idx_ratio = dist_n(8, 4)
            mid_ratio = dist_n(12, 4)

            x_norm = max(0.0, min(1.0, (lms[8].x - ACTIVE_MIN) / (ACTIVE_MAX - ACTIVE_MIN)))
            y_norm = max(0.0, min(1.0, (lms[8].y - ACTIVE_MIN) / (ACTIVE_MAX - ACTIVE_MIN)))
            target_obs_x, target_obs_y = x_norm * SCREEN_W, y_norm * SCREEN_H

            if not (state.two_finger_active or state.scroll_mode):
                max_step = 55.0 if state.dragging else MAX_STEP_PX
                dx = target_obs_x - state.obs_x
                dy = target_obs_y - state.obs_y
                dist = math.hypot(dx, dy)
                if dist > max_step:
                    scale = max_step / dist
                    state.obs_x += dx * scale
                    state.obs_y += dy * scale
                else:
                    state.obs_x, state.obs_y = target_obs_x, target_obs_y

            if now - state.last_ax_query_t > 0.05:
                ax, ay = action_pos()
                if not (state.dragging or state.scroll_mode or state.two_finger_active):
                    state.target_data = snap_targeting.find_actionable_target(ax, ay)
                    state.snapped, (state.snap_x, state.snap_y) = snap_controller.update(ax, ay, state.target_data)
                else:
                    state.snapped = False
                state.last_ax_query_t = now
            
            overlay.update(state.snapped, (state.snap_x, state.snap_y), 
                           state.target_data['bounds'] if state.target_data else None)

            two_finger_pinched = (idx_ratio < TWO_FINGER_ON) and (mid_ratio < TWO_FINGER_ON)
            two_finger_clear   = (idx_ratio > TWO_FINGER_OFF) and (mid_ratio > TWO_FINGER_OFF)
            index_clear = (idx_ratio > (PINCH_OFF + INDEX_CLEAR_MARGIN))
            scroll_pinched = (mid_ratio < MIDDLE_PINCH_ON) and index_clear
            scroll_clear   = (mid_ratio > MIDDLE_PINCH_OFF)
            
            if not state.index_pinched and idx_ratio < PINCH_ON: state.index_pinched = True
            elif state.index_pinched and idx_ratio > PINCH_OFF: state.index_pinched = False

            # ----- GESTURE PRIORITY ENGINE -----
            handled = False
            if now < state.freeze_until:
                mode_str = "FREEZING"
                handled = True
            else:
                # 1. DOUBLE CLICK
                if two_finger_pinched and not state.two_finger_active and (now - state.last_double_t) > 0.42:
                    state.scroll_mode = False
                    if state.dragging: 
                        post_mouse(kCGEventLeftMouseUp, *action_pos())
                        state.dragging = False
                    state.pinch_active = False
                    state.two_finger_active = True
                    double_click(*action_pos())
                    state.last_double_t = now
                    state.freeze_until = now + TWO_FINGER_FREEZE_S
                    mode_str = "DOUBLE CLICK"
                    handled = True
                elif state.two_finger_active:
                    mode_str = "DOUBLE CLICK (HOLD)"
                    if two_finger_clear: state.two_finger_active = False
                    handled = True
            
            if not handled:
                # 2. CLOSED FIST -> MINIMIZE
                busy = state.index_pinched or state.two_finger_active or state.scroll_mode or state.dragging or state.pinch_active
                fist_now = (not busy) and is_closed_fist(lms, hand_size)
                
                if fist_now:
                    if state.fist_start_t is None:
                        state.fist_start_t = now
                    else:
                        held = now - state.fist_start_t
                        if held >= FIST_HOLD_S and (now - state.last_fist_action_t) >= FIST_COOLDOWN_S:
                            send_cmd_m()
                            state.last_fist_action_t = now
                            state.freeze_until = now + 0.25
                            mode_str = "MINIMIZE!"
                            state.fist_start_t = None
                            handled = True
                else:
                    state.fist_start_t = None

                # 3. SCROLL CLUTCH
                if not handled and scroll_pinched and not state.pinch_active and not state.dragging:
                    mode_str = "SCROLLING"
                    if not state.scroll_mode:
                        state.scroll_mode = True
                        state.scroll_start_t = now
                        state.last_mid_y = lms[12].y
                        state.last_scroll_emit = now
                    
                    if (now - state.scroll_start_t) > 0.08:
                        dy = (lms[12].y - state.last_mid_y) * SCREEN_H * SCROLL_MULT
                        if abs(dy) < SCROLL_DEADZONE_PX: dy = 0
                        if dy != 0 and (now - state.last_scroll_emit) >= 0.016:
                            post_scroll(0, -dy)
                            state.last_scroll_emit = now
                    state.last_mid_y = lms[12].y
                    handled = True

                elif state.scroll_mode:
                    if scroll_clear:
                        state.scroll_mode = False
                        if hasattr(state, 'last_mid_y'): del state.last_mid_y

                # 4. SINGLE PINCH (CLICK/DRAG)
                if not handled and state.index_pinched:
                    if not state.pinch_active:
                        state.pinch_active = True
                        state.pinch_start_t = now
                        state.did_drag_this_pinch = False
                        state.drag_release_start = None
                    
                    hold = now - state.pinch_start_t
                    if hold > DRAG_HOLD_S:
                        mode_str = "DRAGGING"
                        if not state.dragging:
                            state.drag_anchor_x, state.drag_anchor_y = action_pos()
                            post_mouse(kCGEventLeftMouseDown, state.drag_anchor_x, state.drag_anchor_y)
                            state.dragging = True
                            state.did_drag_this_pinch = True
                            state.drag_release_start = None
                    else:
                        mode_str = "PINCHING"
                    handled = True
                
                # 5. RELEASE BLOCK
                if not handled:
                    if state.dragging:
                        # Drag release hysteresis
                        if not state.index_pinched:
                            if state.drag_release_start is None:
                                state.drag_release_start = now
                            elif (now - state.drag_release_start) >= DRAG_RELEASE_CONFIRM_S:
                                post_mouse(kCGEventLeftMouseUp, *action_pos())
                                state.dragging = False
                                state.drag_release_start = None
                                state.pinch_active = False
                        else:
                            state.drag_release_start = None
                    else:
                        if state.pinch_active and (not state.did_drag_this_pinch) and (now - state.last_click_t) > 0.13:
                            cx, cy = action_pos()
                            ok = state.snapped and snap_targeting.perform_click(state.target_data)
                            if not ok:
                                post_mouse(kCGEventLeftMouseDown, cx, cy)
                                post_mouse(kCGEventLeftMouseUp, cx, cy)
                            state.last_click_t = now
                        
                        state.pinch_active = False
                        state.did_drag_this_pinch = False
                        state.scroll_mode = False
                        state.two_finger_active = False

        if state.paused: mode_str = "PAUSED"
        hud = [f"MODE: {mode_str}", f"SNAPPED: {state.snapped}", f"Snap d: {snap_controller.last_dist:.1f}px"]
        if state.fist_start_t: hud.append(f"Fist Hold: {now - state.fist_start_t:.1f}s")
        if state.target_data: hud.append(f"Target: {int(state.target_data['bounds'][2])}x{int(state.target_data['bounds'][3])}")
        
        ov_hud = frame.copy()
        cv2.rectangle(ov_hud, (5, 5), (320, 5 + 26*len(hud)+12), (0, 0, 0), -1)
        cv2.addWeighted(ov_hud, 0.45, frame, 0.55, 0, frame)
        for i, line in enumerate(hud): cv2.putText(frame, line, (12, 30+i*26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        
        cv2.imshow('Air Mouse Pro v5.1 Precision+', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    state.running = False
    overlay.cleanup()
    cap.release()
    cv2.destroyAllWindows()
