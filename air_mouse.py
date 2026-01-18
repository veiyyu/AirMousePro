
import cv2
import time
import threading
import sys
import math

# --- ENVIRONMENT CHECK ---
try:
    import numpy as np
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except (ImportError, AttributeError) as e:
    print("-" * 60)
    print(f"FATAL: MediaPipe missing.")
    sys.exit(1)

try:
    from Quartz import (
        CGEventCreateMouseEvent,
        CGEventPost,
        kCGEventMouseMoved,
        kCGEventLeftMouseDown,
        kCGEventLeftMouseUp,
        kCGEventLeftMouseDragged,
        kCGMouseButtonLeft,
        kCGHIDEventTap,
        CGEventCreateScrollWheelEvent,
        kCGScrollEventUnitLine,
        CGEventSetIntegerValueField,
        kCGMouseEventClickState,
        CGEventTapCreate,
        CGEventTapEnable,
        kCGHeadInsertEventTap,
        kCGEventTapOptionDefault,
        kCGEventKeyDown,
        CGEventMaskBit,
        CGEventGetFlags,
        CGEventGetIntegerValueField,
        kCGKeyboardEventKeycode,
        kCGEventFlagMaskCommand,
        kCGEventFlagMaskShift,
        kCGEventFlagMaskControl,
        kCGEventFlagMaskAlternate,
        CFMachPortCreateRunLoopSource,
        CFRunLoopAddSource,
        CFRunLoopGetCurrent,
        kCFRunLoopCommonModes,
        CFRunLoopRun
    )
    from AppKit import NSScreen
except ImportError:
    print("-" * 60)
    print("FATAL: 'pyobjc' missing. Install via: pip install pyobjc")
    sys.exit(1)

# --- CONFIGURATION (STABILITY ENGINE v3.9) ---
WIDTH, HEIGHT = 640, 480
MIRROR = True
CURSOR_HZ = 120 
AB_ALPHA = 0.55 
AB_BETA = 0.02
DRAG_ALPHA = 0.75
FRICTION = 0.88
DRAG_FRICTION = 0.92
STABLE_FRAMES_REQ = 3
LOST_TIMEOUT = 0.2

# Scale-Normalized Pinch Thresholds
PINCH_ON = 0.34
PINCH_OFF = 0.42
INDEX_CLEAR_MARGIN = 0.06
MIDDLE_PINCH_ON = 0.40
MIDDLE_PINCH_OFF = 0.52

# Drag Stability & Prevention
DRAG_PINCH_ON = 0.30  # Requires stronger pinch to initiate drag
DRAG_RELEASE_RATIO = 0.50
DRAG_RELEASE_HOLD_S = 0.15

# State Machine Timing
DRAG_HOLD_S = 0.35
DOUBLE_CLICK_WINDOW_S = 0.70
SINGLE_CLICK_DELAY_S = 0.35

# Click Precision & Nudge
CLICK_LOCK_EXTRA_S = 0.05
CLICK_FREEZE_AFTER_CLICK_S = 0.16 
PRECISION_LOCK_S = 0.08
CLICK_NUDGE_PX = 2
CLICK_NUDGE_DELAY_S = 0.005
ENABLE_CLICK_NUDGE = True

# Scroll Tuning
SCROLL_HZ = 60
SCROLL_GAIN = 0.002
SCROLL_MAX_STEP = 3
SCROLL_DEADZONE = 140
UNPINCH_STABLE_REQ = 1

# --- ACTIVE REGION ---
ACTIVE_MIN = 0.12   
ACTIVE_MAX = 0.88   
EDGE_MARGIN = 0.07  
EDGE_DAMP_FACTOR = 0.25 
SCREEN_SOFT_CLAMP = 40 

# Emergency Pause: Cmd + Ctrl + Opt + P
HOTKEY_KEYCODE = 35 
HOTKEY_FLAGS = kCGEventFlagMaskCommand | kCGEventFlagMaskControl | kCGEventFlagMaskAlternate

# --- INITIALIZATION ---
main_screen = NSScreen.mainScreen()
screen_size = main_screen.frame().size
SCREEN_W = int(screen_size.width)
SCREEN_H = int(screen_size.height)

class AirMouseState:
    def __init__(self):
        self.running = True
        self.paused = False
        self.scroll_mode = False
        self.x, self.y = SCREEN_W // 2, SCREEN_H // 2
        self.vx, self.vy = 0.0, 0.0
        self.tracking_enabled = False
        self.have_obs = False
        self.hand_present_frames = 0
        self.last_hand_seen_time = 0
        self.obs_x, self.obs_y = self.x, self.y
        self.obs_time = None
        self.last_obs_time = None
        self.new_observation = False
        
        # State Machine Fields (Click/Drag)
        self.index_pinched = False
        self.pinch_start_time = None
        self.unpinch_frames = 0
        self.dragging = False
        self.click_anchor_x, self.click_anchor_y = 0.0, 0.0
        self.drag_release_start = None
        
        # Pending Click Dispatcher
        self.pending_single = False
        self.pending_single_time = 0.0
        self.pending_anchor_x = 0.0
        self.pending_anchor_y = 0.0

        # Scroll Fields
        self.v_s = 0.0
        self.scroll_accum = 0.0
        self.prev_center_y = None
        self.prev_center_t = None
        self.last_scroll_emit = 0.0
        
        # Precision Locks
        self.pointer_freeze_until = 0.0
        self.click_lock_active = False
        self.click_lock_until = 0.0

state = AirMouseState()

def reset_gestures():
    state.index_pinched = False
    state.unpinch_frames = 0
    state.dragging = False
    state.scroll_mode = False
    state.v_s = 0.0
    state.scroll_accum = 0.0
    state.vx = 0.0
    state.vy = 0.0
    state.new_observation = False
    state.last_obs_time = None
    state.prev_center_y = None
    state.prev_center_t = None
    state.pointer_freeze_until = 0.0
    state.click_lock_until = 0.0
    state.click_lock_active = False
    state.pinch_start_time = None
    state.drag_release_start = None
    state.pending_single = False

def post_mouse_event(event_type, x, y, button=kCGMouseButtonLeft, click_count=1):
    if state.paused: return
    x = max(0, min(x, SCREEN_W - 1))
    y = max(0, min(y, SCREEN_H - 1))
    event = CGEventCreateMouseEvent(None, event_type, (x, y), button)
    CGEventSetIntegerValueField(event, kCGMouseEventClickState, click_count)
    CGEventPost(kCGHIDEventTap, event)

def single_click(x, y):
    post_mouse_event(kCGEventLeftMouseDown, x, y, click_count=1)
    post_mouse_event(kCGEventLeftMouseUp,   x, y, click_count=1)

def single_click_precise(x, y):
    if not ENABLE_CLICK_NUDGE:
        single_click(x, y)
        return

    dx = CLICK_NUDGE_PX
    post_mouse_event(kCGEventMouseMoved, x + dx, y)
    time.sleep(CLICK_NUDGE_DELAY_S)
    post_mouse_event(kCGEventMouseMoved, x - dx, y)
    time.sleep(CLICK_NUDGE_DELAY_S)
    post_mouse_event(kCGEventMouseMoved, x, y)
    time.sleep(CLICK_NUDGE_DELAY_S)

    post_mouse_event(kCGEventLeftMouseDown, x, y, click_count=1)
    post_mouse_event(kCGEventLeftMouseUp,   x, y, click_count=1)

def double_click(x, y):
    post_mouse_event(kCGEventLeftMouseDown, x, y, click_count=1)
    post_mouse_event(kCGEventLeftMouseUp,   x, y, click_count=1)
    time.sleep(0.01)
    post_mouse_event(kCGEventLeftMouseDown, x, y, click_count=2)
    post_mouse_event(kCGEventLeftMouseUp,   x, y, click_count=2)

# --- EMERGENCY STOP ---
def event_tap_callback(proxy, type_, event, refcon):
    if type_ == kCGEventKeyDown:
        keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
        flags = CGEventGetFlags(event)
        if keycode == HOTKEY_KEYCODE and (flags & HOTKEY_FLAGS) == HOTKEY_FLAGS:
            state.paused = not state.paused
            if state.paused: reset_gestures()
            return None
    return event

def hotkey_listener():
    event_mask = CGEventMaskBit(kCGEventKeyDown)
    tap = CGEventTapCreate(kCGHIDEventTap, kCGHeadInsertEventTap, kCGEventTapOptionDefault, event_mask, event_tap_callback, None)
    if not tap: return
    source = CFMachPortCreateRunLoopSource(None, tap, 0)
    CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes)
    CGEventTapEnable(tap, True)
    CFRunLoopRun()

threading.Thread(target=hotkey_listener, daemon=True).start()

# --- CURSOR THREAD (120Hz) ---
def cursor_thread():
    dt = 1.0 / CURSOR_HZ
    while state.running:
        start_t = time.perf_counter()
        now = start_t
        if not state.paused and state.have_obs:
            if not state.scroll_mode:
                is_click_locked = (state.click_lock_active or now < state.click_lock_until) and not state.dragging
                
                if is_click_locked:
                    state.vx = 0
                    state.vy = 0
                    state.x = state.click_anchor_x
                    state.y = state.click_anchor_y
                    post_mouse_event(kCGEventMouseMoved, state.x, state.y)
                else:
                    state.x += state.vx * dt
                    state.y += state.vy * dt
                    
                    cur_friction = DRAG_FRICTION if state.dragging else FRICTION
                    cur_alpha = DRAG_ALPHA if state.dragging else AB_ALPHA

                    if state.x < SCREEN_SOFT_CLAMP or state.x > SCREEN_W - SCREEN_SOFT_CLAMP:
                        state.vx *= 0.55
                    if state.y < SCREEN_SOFT_CLAMP or state.y > SCREEN_H - SCREEN_SOFT_CLAMP:
                        state.vy *= 0.55

                    state.vx *= cur_friction
                    state.vy *= cur_friction

                    if state.new_observation:
                        dx = state.obs_x - state.x
                        dy = state.obs_y - state.y
                        dt_obs = state.obs_time - (state.last_obs_time or (state.obs_time - 0.033))
                        state.last_obs_time = state.obs_time
                        dt_obs = max(1/120, min(dt_obs, 1/15))

                        state.x += cur_alpha * dx
                        state.y += cur_alpha * dy

                        if abs(dx) < 140 and abs(dy) < 140:
                            state.vx += (AB_BETA / dt_obs) * dx
                            state.vy += (AB_BETA / dt_obs) * dy
                        state.new_observation = False

                    state.x = max(0, min(state.x, SCREEN_W - 1))
                    state.y = max(0, min(state.y, SCREEN_H - 1))
                    
                    evt_type = kCGEventLeftMouseDragged if state.dragging else kCGEventMouseMoved
                    post_mouse_event(evt_type, state.x, state.y)
            
            elif abs(state.v_s) > SCROLL_DEADZONE:
                if now - state.last_scroll_emit >= (1.0 / SCROLL_HZ):
                    step_float = max(min(SCROLL_GAIN * state.v_s, SCROLL_MAX_STEP), -SCROLL_MAX_STEP)
                    state.scroll_accum += step_float
                    state.scroll_accum *= 0.90
                    emit = int(state.scroll_accum)
                    if emit != 0:
                        event = CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitLine, 1, emit)
                        CGEventPost(kCGHIDEventTap, event)
                        state.scroll_accum -= emit
                    state.last_scroll_emit = now

        time.sleep(max(0, dt - (time.perf_counter() - start_t)))

threading.Thread(target=cursor_thread, daemon=True).start()

# --- MAIN INFERENCE LOOP ---
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

try:
    detector = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        now = time.perf_counter()
        frame = cv2.flip(frame, 1) if MIRROR else frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        rgb.flags.writeable = False
        res = detector.process(rgb)
        rgb.flags.writeable = True

        if res.multi_hand_landmarks:
            state.last_hand_seen_time = now
            state.hand_present_frames += 1
            if state.hand_present_frames >= STABLE_FRAMES_REQ:
                state.tracking_enabled = True
            
            for hand_lms in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                lms = hand_lms.landmark
                
                hand_size = math.sqrt((lms[5].x - lms[17].x)**2 + (lms[5].y - lms[17].y)**2) or 0.01
                index_dist = math.sqrt((lms[8].x - lms[4].x)**2 + (lms[8].y - lms[4].y)**2)
                middle_dist = math.sqrt((lms[12].x - lms[4].x)**2 + (lms[12].y - lms[4].y)**2)
                
                index_pinch_ratio = index_dist / hand_size
                middle_pinch_ratio = middle_dist / hand_size
                index_clear = (index_pinch_ratio > PINCH_OFF + INDEX_CLEAR_MARGIN)

                if middle_pinch_ratio < MIDDLE_PINCH_ON and index_clear:
                    if not state.scroll_mode:
                        if state.dragging:
                            post_mouse_event(kCGEventLeftMouseUp, state.x, state.y, click_count=1)
                        state.dragging = False
                        state.index_pinched = False
                        state.pinch_start_time = None
                        state.pending_single = False
                        state.click_lock_active = False
                    state.scroll_mode = True
                elif middle_pinch_ratio > MIDDLE_PINCH_OFF:
                    state.scroll_mode = False

                x_norm = max(0, min((lms[8].x - ACTIVE_MIN) / (ACTIVE_MAX - ACTIVE_MIN), 1))
                y_norm = max(0, min((lms[8].y - ACTIVE_MIN) / (ACTIVE_MAX - ACTIVE_MIN), 1))
                new_obs_x, new_obs_y = x_norm * SCREEN_W, y_norm * SCREEN_H
                
                if not (now < state.pointer_freeze_until or state.click_lock_active or (state.index_pinched and not state.dragging)):
                    if (lms[8].x < EDGE_MARGIN or lms[8].x > (1-EDGE_MARGIN) or lms[8].y < EDGE_MARGIN or lms[8].y > (1-EDGE_MARGIN)):
                        state.obs_x += (new_obs_x - state.obs_x) * EDGE_DAMP_FACTOR
                        state.obs_y += (new_obs_y - state.obs_y) * EDGE_DAMP_FACTOR
                    else:
                        state.obs_x, state.obs_y = new_obs_x, new_obs_y
                    state.obs_time = now
                    state.new_observation = True
                    state.have_obs = True

                if not state.scroll_mode:
                    if not state.index_pinched and index_pinch_ratio < PINCH_ON:
                        state.index_pinched = True
                        state.pinch_start_time = now
                        state.dragging = False
                        state.unpinch_frames = 0
                        state.click_lock_active = True
                        state.vx = 0
                        state.vy = 0
                        state.click_anchor_x, state.click_anchor_y = state.x, state.y
                    
                    elif state.index_pinched:
                        if state.dragging:
                            if index_pinch_ratio > DRAG_RELEASE_RATIO:
                                if state.drag_release_start is None:
                                    state.drag_release_start = now
                                elif (now - state.drag_release_start) >= DRAG_RELEASE_HOLD_S:
                                    post_mouse_event(kCGEventLeftMouseUp, state.x, state.y, click_count=1)
                                    state.dragging = False
                                    state.index_pinched = False
                                    state.click_lock_active = False
                                    state.click_lock_until = now + CLICK_LOCK_EXTRA_S
                            else:
                                state.drag_release_start = None
                        else:
                            if index_pinch_ratio > PINCH_OFF:
                                state.unpinch_frames += 1
                            else:
                                state.unpinch_frames = 0

                            if state.unpinch_frames >= UNPINCH_STABLE_REQ:
                                state.index_pinched = False
                                state.click_lock_active = False
                                dt_since_last_pending = now - state.pending_single_time
                                if state.pending_single and dt_since_last_pending <= DOUBLE_CLICK_WINDOW_S:
                                    double_click(state.pending_anchor_x, state.pending_anchor_y)
                                    state.pending_single = False
                                    state.click_lock_until = now + CLICK_LOCK_EXTRA_S
                                    state.pointer_freeze_until = now + CLICK_FREEZE_AFTER_CLICK_S
                                else:
                                    state.pending_single = True
                                    state.pending_single_time = now
                                    state.pending_anchor_x, state.pending_anchor_y = state.click_anchor_x, state.click_anchor_y
                                    state.click_lock_until = now + CLICK_LOCK_EXTRA_S
                                
                                state.pinch_start_time = None

                        if state.index_pinched and not state.dragging:
                            # TRIGGER DRAG (Modified v3.9)
                            if (now - state.pinch_start_time >= DRAG_HOLD_S) and (index_pinch_ratio < DRAG_PINCH_ON):
                                post_mouse_event(kCGEventLeftMouseDown, state.click_anchor_x, state.click_anchor_y, click_count=1)
                                state.dragging = True
                                state.click_lock_active = False
                                state.drag_release_start = None
                                state.pending_single = False

                if state.scroll_mode:
                    center_y = (lms[0].y + lms[9].y) / 2.0 * SCREEN_H
                    if state.prev_center_y is not None and state.prev_center_t is not None:
                        dt_s = max(0.01, now - state.prev_center_t)
                        v_raw = (center_y - state.prev_center_y) / dt_s
                        state.v_s = 0.12 * v_raw + 0.88 * state.v_s
                    state.prev_center_y, state.prev_center_t = center_y, now
                else:
                    state.v_s = 0.0
                    state.prev_center_y, state.prev_center_t = None, None

            # Pending Single-Click Dispatcher (Modified v3.9: Precision Snap)
            if state.pending_single:
                if (now - state.pending_single_time) > SINGLE_CLICK_DELAY_S:
                    # Apply Precision Snap before firing
                    state.pointer_freeze_until = now + PRECISION_LOCK_S
                    state.vx = 0
                    state.vy = 0
                    state.x = state.pending_anchor_x
                    state.y = state.pending_anchor_y
                    post_mouse_event(kCGEventMouseMoved, state.x, state.y)
                    
                    time.sleep(PRECISION_LOCK_S)
                    single_click_precise(state.pending_anchor_x, state.pending_anchor_y)
                    
                    state.pending_single = False
                    state.pointer_freeze_until = time.perf_counter() + CLICK_FREEZE_AFTER_CLICK_S

        else:
            state.hand_present_frames = 0
            if now - state.last_hand_seen_time > LOST_TIMEOUT:
                state.tracking_enabled = False
                reset_gestures()

        status = "PAUSED" if state.paused else ("NO HAND" if not state.tracking_enabled else ("SCROLLING" if state.scroll_mode else ("DRAGGING" if state.dragging else "READY")))
        color = (0, 0, 255) if state.paused or not state.tracking_enabled else (0, 255, 0)
        cv2.putText(frame, f"v3.9 | {status}", (20, 40), 1, 1.4, color, 2)
        if state.index_pinched and not state.dragging:
            progress = min(1.0, (now - state.pinch_start_time) / DRAG_HOLD_S)
            cv2.rectangle(frame, (20, 60), (20 + int(progress*150), 75), (255, 255, 0), -1)
        if state.pending_single:
            cv2.circle(frame, (20, 90), 5, (255, 0, 255), -1)
            
        cv2.imshow('Air Mouse Pro', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    state.running = False
    cap.release()
    cv2.destroyAllWindows()
