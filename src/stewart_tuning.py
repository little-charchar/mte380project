import cv2
import numpy as np
import json
import serial
import time
import math
from threading import Thread
import queue

# =========================================================
# TUNING SETTINGS
# =========================================================
USB_PORT = "/dev/cu.usbmodem1401"   # Run `ls /dev/cu.usb*` in terminal to check
BAUD_RATE = 115200
CAMERA_INDEX = 0

# 1. Neutral & Leveling Trim
PHYSICAL_NEUTRAL = 70.0             # Must match NEUTRAL_ANGLE in Arduino
SERVO_TRIM = [8.0, 0.0, 0.0]        # Nudge by +/- 0.5 to 4.0 deg if one corner is low when flat
MIN_SERVO_ANGLE = 55                # Must match MIN_SAFE_ANGLE in Arduino
MAX_SERVO_ANGLE = 85                # Must match MAX_SAFE_ANGLE in Arduino

# 2. Starting PD Gains (Also adjustable live via OpenCV sliders or 't' Twiddle tuner)
DEFAULT_KP = 0.45
DEFAULT_KI = 0.0
DEFAULT_KD = 0.55
MAX_TILT_DEG = 2.5

DEADBAND_CM = 0.60                  # Capture radius around center (also adjustable via slider)
MIN_ANGLE_CHANGE = 0.0              # Allow smooth 0.2 deg micro-adjustments without step jumps

# 3. Mouse-Drawn Trajectory Settings
TRAJ_SPEED_CM_S = 2.0               # Speed the target moves along your drawn line (cm/s)
TRAJ_RAMP_SEC = 1.5                 # Seconds to smoothly blend from current ball pos onto the path
KV_FEEDFORWARD = 0.12               # Gentle tilt (deg) per (cm/s) along path tangent

# 4. Axis Direction Signs (Flip to -1.0 or 1.0 if an axis tilts the wrong way)
SIGN_Y_AXIS = -1.0                  # Controls front/back tilt (from camera Y)
SIGN_X_AXIS = -1.0                  # Controls left/right tilt (from camera X)

# 5. Ball Color & Size Detection (HSV)
BALL_HSV_LOWER = np.array([5, 150, 120])
BALL_HSV_UPPER = np.array([25, 255, 255])
MIN_BALL_RADIUS = 15                # Pixels (increase if background specks trigger false circles)
# =========================================================


# ---------------------------------------------------------
# Kinematics Module
# ---------------------------------------------------------
class Plate3DoFKinematics:
    def __init__(self, l_base=0.080, l_crank=0.030, l_rod=0.120, r_plat=0.070, nominal_height=0.115):
        self.l1 = self.l2 = self.l3 = l_base
        self.l11 = self.l21 = self.l31 = l_crank
        self.l12 = self.l22 = self.l32 = l_rod
        self.r_plat = r_plat
        self.z0 = nominal_height

        self.p1_local = np.array([0.0, self.r_plat, 0.0])
        self.p2_local = np.array([math.cos(math.radians(30)) * self.r_plat, -math.sin(math.radians(30)) * self.r_plat, 0.0])
        self.p3_local = np.array([-math.cos(math.radians(30)) * self.r_plat, -math.sin(math.radians(30)) * self.r_plat, 0.0])

    def get_vertex_positions(self, tilt_x_rad, tilt_y_rad, heave_m=0.0):
        cx, sx = math.cos(tilt_x_rad), math.sin(tilt_x_rad)
        cy, sy = math.cos(tilt_y_rad), math.sin(tilt_y_rad)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        R = Rx @ Ry

        origin = np.array([0.0, 0.0, self.z0 + heave_m])
        x1 = origin + R @ self.p1_local
        x2 = origin + R @ self.p2_local
        x3 = origin + R @ self.p3_local
        return x1, x2, x3

    def calculate_ik_actuator_1(self, x1):
        y1, z1 = x1[1], x1[2]
        AA_tmp = self.l12**2 - (self.l1 - y1)**2 - self.l11**2 - z1**2
        bb_tmp = 2.0 * (self.l1 - y1) * self.l11
        cc_tmp = 2.0 * z1 * self.l11
        denom_AA = math.sqrt(bb_tmp**2 + cc_tmp**2)

        CC = math.acos(np.clip(AA_tmp / denom_AA, -1.0, 1.0))
        bb1 = math.acos(np.clip(bb_tmp / denom_AA, -1.0, 1.0))
        theta_11 = CC - bb1

        sin_arg = np.clip((z1 - self.l11 * math.sin(theta_11)) / self.l12, -1.0, 1.0)
        theta_12 = math.asin(sin_arg)
        return math.degrees(theta_11), math.degrees(theta_12)

    def calculate_ik_actuator_2(self, x2):
        y_tmp = math.sqrt(x2[0]**2 + x2[1]**2)
        z_tmp = x2[2]

        AA_tmp = -(self.l22**2 - (self.l2 - y_tmp)**2 - self.l21**2 - z_tmp**2)
        bb_tmp = 2.0 * (self.l2 - y_tmp) * self.l21
        cc_tmp = 2.0 * z_tmp * self.l21
        denom_AA = math.sqrt(bb_tmp**2 + cc_tmp**2)

        CC = math.asin(np.clip(AA_tmp / denom_AA, -1.0, 1.0))
        bb1 = math.asin(np.clip(bb_tmp / denom_AA, -1.0, 1.0))
        theta_21 = CC + bb1

        sin_arg = np.clip((z_tmp - self.l21 * math.sin(theta_21)) / self.l22, -1.0, 1.0)
        theta_22 = math.asin(sin_arg)
        return math.degrees(theta_21), math.degrees(theta_22)

    def calculate_ik_actuator_3(self, x3):
        y_tmp = math.sqrt(x3[0]**2 + x3[1]**2)
        z_tmp = x3[2]

        AA_tmp = self.l32**2 - (self.l3 - y_tmp)**2 - self.l31**2 - z_tmp**2
        bb_tmp = 2.0 * (self.l3 - y_tmp) * self.l31
        cc_tmp = 2.0 * z_tmp * self.l31
        denom_AA = math.sqrt(bb_tmp**2 + cc_tmp**2)

        CC = math.acos(np.clip(AA_tmp / denom_AA, -1.0, 1.0))
        bb2 = math.asin(np.clip(cc_tmp / denom_AA, -1.0, 1.0))
        theta_31 = CC - bb2

        sin_arg = np.clip((z_tmp - self.l31 * math.sin(theta_31)) / self.l32, -1.0, 1.0)
        theta_32 = math.asin(sin_arg)
        return math.degrees(theta_31), math.degrees(theta_32)

    def solve(self, tilt_x_rad, tilt_y_rad):
        x1, x2, x3 = self.get_vertex_positions(tilt_x_rad, tilt_y_rad)
        th11, _ = self.calculate_ik_actuator_1(x1)
        th21, _ = self.calculate_ik_actuator_2(x2)
        th31, _ = self.calculate_ik_actuator_3(x3)

        NEUTRAL_IK = 3.5
        cmd_1 = PHYSICAL_NEUTRAL + SERVO_TRIM[0] + (th11 - NEUTRAL_IK)
        cmd_2 = PHYSICAL_NEUTRAL + SERVO_TRIM[1] + (th21 - NEUTRAL_IK)
        cmd_3 = PHYSICAL_NEUTRAL + SERVO_TRIM[2] + (th31 - NEUTRAL_IK)
        return [cmd_1, cmd_2, cmd_3]


# ---------------------------------------------------------
# Ball Detection Helper (OpenCV)
# ---------------------------------------------------------
def detect_ball_xy(frame):
    vis_frame = frame.copy()
    h, w = frame.shape[:2]
    mid_x, mid_y = int(w / 2), int(h / 2)

    # Permanent red center dot & crosshair
    cv2.circle(vis_frame, (mid_x, mid_y), 5, (0, 0, 255), -1)
    cv2.drawMarker(vis_frame, (mid_x, mid_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        ((_, _), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        if radius > MIN_BALL_RADIUS and M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            cv2.circle(vis_frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"BALL DETECTED (r={int(radius)})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            norm_x = (cx - mid_x) / (w / 2.0)
            norm_y = (cy - mid_y) / (h / 2.0)
            return True, norm_x, norm_y, vis_frame

    cv2.putText(vis_frame, "NO BALL (HOLDING LEVEL)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return False, 0.0, 0.0, vis_frame


# ---------------------------------------------------------
# Main Controller: Anti-Oscillation PD + Trajectory + Twiddle
# ---------------------------------------------------------
class StewartController:
    def __init__(self, config_file="config.json"):
        self.config = {
            "servo": {"port": USB_PORT, "baud": BAUD_RATE},
            "camera": {"index": CAMERA_INDEX, "frame_width": 640, "frame_height": 480},
            "calibration": {"pixel_to_meter_ratio": 0.0004}
        }
        try:
            with open(config_file, 'r') as f:
                self.config.update(json.load(f))
        except FileNotFoundError:
            pass

        self.ik = Plate3DoFKinematics()

        self.Kp = DEFAULT_KP
        self.Ki = DEFAULT_KI
        self.Kd = DEFAULT_KD
        self.deadband_cm = DEADBAND_CM

        self.smooth_pos_x = None
        self.smooth_pos_y = None
        self.last_sent_angles = None
        self.deadband_locked = False
        self.current_err_cm = 0.0

        self.scale_factor_x = self.config['calibration']['pixel_to_meter_ratio'] * (self.config['camera']['frame_width'] / 2)
        self.scale_factor_y = self.config['calibration']['pixel_to_meter_ratio'] * (self.config['camera']['frame_height'] / 2)

        self.ser = None
        self.setpoint_x = 0.0
        self.setpoint_y = 0.0

        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_pos_x = None
        self.prev_pos_y = None
        self.vel_filtered_x = 0.0
        self.vel_filtered_y = 0.0

        # Mouse-drawn trajectory state
        self.custom_waypoints = None
        self.custom_closed = False
        self.traj_start_time = 0.0
        self.traj_start_pos = (0.0, 0.0)
        self.drawn_points_px = []
        self.is_drawing = False

        # Twiddle auto-tuner state
        self.twiddle_active = False
        self.twiddle_running_trial = False
        self.twiddle_trial_start = 0.0
        self.twiddle_cost = 0.0
        self.twiddle_p = [DEFAULT_KP, DEFAULT_KD]
        self.twiddle_dp = [0.15, 0.15]
        self.twiddle_best_p = [DEFAULT_KP, DEFAULT_KD]
        self.twiddle_best_err = float("inf")
        self.twiddle_idx = 0
        self.twiddle_state = 0
        self.twiddle_p_base = DEFAULT_KP
        self.twiddle_status = ""
        self.twiddle_ready = True

        self.position_queue = queue.Queue(maxsize=1)
        self.running = False
        self.latest_vis_frame = None

    def connect_serial(self):
        try:
            self.ser = serial.Serial(self.config['servo']['port'], self.config['servo']['baud'], timeout=0.05)
            time.sleep(2.0)
            print(f"[SERVO] Connected to Arduino Mega on {self.config['servo']['port']}")
            return True
        except Exception as e:
            print(f"[SERVO] Serial error: {e}. Running in simulation mode.")
            return False

    def send_angles(self, theta_list, force=False):
        if self.ser and self.ser.is_open:
            clamped = [float(np.clip(th, MIN_SERVO_ANGLE, MAX_SERVO_ANGLE)) for th in theta_list]

            if not force and MIN_ANGLE_CHANGE > 0.0 and self.last_sent_angles is not None:
                max_diff = max(abs(a - b) for a, b in zip(clamped, self.last_sent_angles))
                if max_diff < MIN_ANGLE_CHANGE:
                    return

            self.last_sent_angles = clamped
            encoded = [int(np.clip(round((a - 45.0) * 5.0), 0, 250)) for a in clamped]
            packet = bytearray([0xFF] + encoded)
            try:
                self.ser.write(packet)
            except Exception:
                pass

    # -----------------------------------------------------
    # Trajectory Helpers
    # -----------------------------------------------------
    def set_custom_trajectory(self, waypoints_m, closed=True):
        if self.twiddle_active:
            self.toggle_twiddle()
        self.custom_closed = closed
        self.traj_start_time = time.time()
        sx = self.smooth_pos_x if self.smooth_pos_x is not None else 0.0
        sy = self.smooth_pos_y if self.smooth_pos_y is not None else 0.0
        self.traj_start_pos = (sx, sy)
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.deadband_locked = False
        self.custom_waypoints = np.array(waypoints_m, dtype=float)
        print(f"[TRAJ] Custom trajectory active ({len(waypoints_m)} pts, closed={closed})")

    def clear_trajectory(self):
        self.custom_waypoints = None
        self.drawn_points_px = []
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.deadband_locked = False
        print("[TRAJ] Path cleared - returning ball to center")

    def get_trajectory_target(self):
        pts = self.custom_waypoints
        if pts is None or len(pts) < 2:
            return self.setpoint_x, self.setpoint_y, 0.0, 0.0

        path = np.vstack([pts, pts[0]]) if self.custom_closed else pts
        seg_lengths = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))
        cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_len = cum_dist[-1]

        if total_len < 0.0005:
            return float(path[0, 0]), float(path[0, 1]), 0.0, 0.0

        speed_m_s = TRAJ_SPEED_CM_S * 0.01
        elapsed = time.time() - self.traj_start_time
        dist = (elapsed * speed_m_s) % total_len if self.custom_closed else min(elapsed * speed_m_s, total_len)

        raw_x = float(np.interp(dist, cum_dist, path[:, 0]))
        raw_y = float(np.interp(dist, cum_dist, path[:, 1]))

        ds = 0.0005
        d_next = (dist + ds) % total_len if self.custom_closed else min(dist + ds, total_len)
        nx = float(np.interp(d_next, cum_dist, path[:, 0]))
        ny = float(np.interp(d_next, cum_dist, path[:, 1]))
        step = np.hypot(nx - raw_x, ny - raw_y)

        if step > 1e-6 and (self.custom_closed or dist < total_len - ds):
            raw_vx_cms = TRAJ_SPEED_CM_S * (nx - raw_x) / step
            raw_vy_cms = TRAJ_SPEED_CM_S * (ny - raw_y) / step
        else:
            raw_vx_cms, raw_vy_cms = 0.0, 0.0

        blend = min(1.0, elapsed / TRAJ_RAMP_SEC)
        sx, sy = self.traj_start_pos
        target_x_m = (1.0 - blend) * sx + blend * raw_x
        target_y_m = (1.0 - blend) * sy + blend * raw_y
        target_vx_cms = blend * raw_vx_cms
        target_vy_cms = blend * raw_vy_cms

        return target_x_m, target_y_m, target_vx_cms, target_vy_cms

    # -----------------------------------------------------
    # Twiddle Auto-Tuner Helpers
    # -----------------------------------------------------
    def toggle_twiddle(self):
        self.twiddle_active = not self.twiddle_active
        self.twiddle_running_trial = False
        self.twiddle_ready = True
        if self.twiddle_active:
            self.clear_trajectory()
            self.Ki = 0.0
            self.twiddle_p = [max(0.1, self.Kp), max(0.1, self.Kd)]
            self.twiddle_dp = [0.15, 0.15]
            self.twiddle_best_p = list(self.twiddle_p)
            self.twiddle_best_err = float("inf")
            self.twiddle_idx = 0
            self.twiddle_state = 0
            self.twiddle_p_base = self.twiddle_p[0]
            self.twiddle_status = "TWIDDLE ON: Nudge ball >2cm to start trial 1"
            print(f"[TWIDDLE] Started from Kp={self.Kp:.2f}, Kd={self.Kd:.2f} (Ki locked to 0)")
        else:
            self.Kp, self.Kd = self.twiddle_best_p
            self.twiddle_status = ""
            print(f"[TWIDDLE] Stopped. Best gains applied: Kp={self.Kp:.2f}, Kd={self.Kd:.2f}")

    def _twiddle_step(self, err):
        i = self.twiddle_idx
        p, dp = self.twiddle_p, self.twiddle_dp
        print(f"[TWIDDLE] Trial (Kp={p[0]:.2f}, Kd={p[1]:.2f}) -> Cost={err:.2f} (Best={self.twiddle_best_err:.2f})")

        if self.twiddle_state == 0:
            self.twiddle_best_err = err
            self.twiddle_best_p = list(p)
            self.twiddle_p_base = p[i]
            p[i] = round(self.twiddle_p_base + dp[i], 3)
            self.twiddle_state = 1

        elif self.twiddle_state == 1:
            if err < self.twiddle_best_err:
                self.twiddle_best_err = err
                self.twiddle_best_p = list(p)
                dp[i] *= 1.1
                self.twiddle_idx = (i + 1) % 2
                ni = self.twiddle_idx
                self.twiddle_p_base = p[ni]
                p[ni] = round(self.twiddle_p_base + dp[ni], 3)
                self.twiddle_state = 1
            else:
                p[i] = round(max(0.1, self.twiddle_p_base - dp[i]), 3)
                self.twiddle_state = 2

        elif self.twiddle_state == 2:
            if err < self.twiddle_best_err:
                self.twiddle_best_err = err
                self.twiddle_best_p = list(p)
                dp[i] *= 1.1
            else:
                p[i] = self.twiddle_p_base
                dp[i] *= 0.9

            self.twiddle_idx = (i + 1) % 2
            ni = self.twiddle_idx
            self.twiddle_p_base = p[ni]
            p[ni] = round(self.twiddle_p_base + dp[ni], 3)
            self.twiddle_state = 1

        self.Kp, self.Kd = p[0], p[1]
        self.twiddle_status = f"Next: Kp={self.Kp:.2f}, Kd={self.Kd:.2f} | Best={self.twiddle_best_err:.1f}"

    # -----------------------------------------------------
    # Smooth Linear PD Controller with Hysteresis Deadband
    # -----------------------------------------------------
    def update_pid_2d(self, pos_x, pos_y, dt=0.033):
        if self.smooth_pos_x is None:
            self.smooth_pos_x, self.smooth_pos_y = pos_x, pos_y
        else:
            self.smooth_pos_x = 0.85 * pos_x + 0.15 * self.smooth_pos_x
            self.smooth_pos_y = 0.85 * pos_y + 0.15 * self.smooth_pos_y

        target_x_m, target_y_m, target_vx_cms, target_vy_cms = self.get_trajectory_target()
        is_tracking_path = self.custom_waypoints is not None

        err_x = (target_x_m - self.smooth_pos_x) * 100.0
        err_y = (target_y_m - self.smooth_pos_y) * 100.0

        if self.prev_pos_x is None or self.prev_pos_y is None:
            raw_vx, raw_vy = 0.0, 0.0
        else:
            raw_vx = ((self.smooth_pos_x - self.prev_pos_x) * 100.0) / dt
            raw_vy = ((self.smooth_pos_y - self.prev_pos_y) * 100.0) / dt

        self.prev_pos_x = self.smooth_pos_x
        self.prev_pos_y = self.smooth_pos_y

        self.vel_filtered_x = 0.60 * raw_vx + 0.40 * self.vel_filtered_x
        self.vel_filtered_y = 0.60 * raw_vy + 0.40 * self.vel_filtered_y

        dist_cm = math.hypot(err_x, err_y)
        self.current_err_cm = dist_cm
        speed_cms = math.hypot(self.vel_filtered_x, self.vel_filtered_y)

        # Hysteresis deadband: snaps plate flat when ball slows over crosshair,
        # and refuses to twitch until ball drifts past 1.4x deadband_cm
        if not is_tracking_path and self.deadband_cm > 0.0:
            if self.deadband_locked:
                if dist_cm < self.deadband_cm * 1.4:
                    self.integral_x = 0.0
                    self.integral_y = 0.0
                    return 0.0, 0.0
                self.deadband_locked = False
            else:
                if dist_cm < self.deadband_cm and speed_cms < 3.0:
                    self.deadband_locked = True
                    self.integral_x = 0.0
                    self.integral_y = 0.0
                    return 0.0, 0.0

            if dist_cm < self.deadband_cm * 2.0:
                scale = max(0.0, (dist_cm - self.deadband_cm) / self.deadband_cm)
                err_x *= scale
                err_y *= scale
        else:
            self.deadband_locked = False

        vx, vy = self.vel_filtered_x, self.vel_filtered_y
        if is_tracking_path:
            damped_vx = self.Kd * (vx - 0.5 * target_vx_cms)
            damped_vy = self.Kd * (vy - 0.5 * target_vy_cms)
            ff_x_deg = KV_FEEDFORWARD * target_vx_cms
            ff_y_deg = KV_FEEDFORWARD * target_vy_cms
        else:
            damped_vx = self.Kd * vx
            damped_vy = self.Kd * vy
            ff_x_deg, ff_y_deg = 0.0, 0.0

        self.integral_x = np.clip(self.integral_x + err_x * dt, -10.0, 10.0)
        self.integral_y = np.clip(self.integral_y + err_y * dt, -10.0, 10.0)

        tilt_x_deg = SIGN_Y_AXIS * (self.Kp * err_y + self.Ki * self.integral_y - damped_vy + ff_y_deg)
        tilt_y_deg = SIGN_X_AXIS * (self.Kp * err_x + self.Ki * self.integral_x - damped_vx + ff_x_deg)

        tilt_x_rad = math.radians(np.clip(tilt_x_deg, -MAX_TILT_DEG, MAX_TILT_DEG))
        tilt_y_rad = math.radians(np.clip(tilt_y_deg, -MAX_TILT_DEG, MAX_TILT_DEG))
        return tilt_x_rad, tilt_y_rad

    # -----------------------------------------------------
    # Mouse & Threads
    # -----------------------------------------------------
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_drawing = False
            self.clear_trajectory()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.drawn_points_px = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            if np.hypot(x - self.drawn_points_px[-1][0], y - self.drawn_points_px[-1][1]) > 8:
                self.drawn_points_px.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP and self.is_drawing:
            self.is_drawing = False
            if len(self.drawn_points_px) >= 5:
                w = self.config['camera']['frame_width']
                h = self.config['camera']['frame_height']
                pts_m = [
                    (((px - w / 2.0) / (w / 2.0)) * self.scale_factor_x,
                     ((py - h / 2.0) / (h / 2.0)) * self.scale_factor_y)
                    for px, py in self.drawn_points_px
                ]
                closed = np.hypot(pts_m[-1][0] - pts_m[0][0], pts_m[-1][1] - pts_m[0][1]) < 0.015
                self.set_custom_trajectory(pts_m, closed=closed)
            else:
                self.clear_trajectory()

    def camera_thread(self):
        cap = cv2.VideoCapture(self.config['camera']['index'], cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['frame_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['frame_height'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            found, x_norm, y_norm, vis_frame = detect_ball_xy(frame)
            payload = (x_norm * self.scale_factor_x, y_norm * self.scale_factor_y) if found else (None, None)

            try:
                if self.position_queue.full():
                    self.position_queue.get_nowait()
                self.position_queue.put_nowait(payload)
            except Exception:
                pass

            self.latest_vis_frame = vis_frame

        cap.release()

    def control_thread(self):
        self.connect_serial()
        last_t = time.time()

        while self.running:
            try:
                pos_x, pos_y = self.position_queue.get(timeout=0.1)
                now = time.time()
                dt = max(now - last_t, 0.001)
                last_t = now

                if pos_x is None or pos_y is None:
                    if self.twiddle_active:
                        if self.twiddle_running_trial:
                            self.twiddle_running_trial = False
                            self._twiddle_step(999.0)
                        self.twiddle_ready = True  # Lifting the ball off arms the next trial

                    self.integral_x = 0.0
                    self.integral_y = 0.0
                    self.prev_pos_x = None
                    self.prev_pos_y = None
                    self.smooth_pos_x = None
                    self.smooth_pos_y = None
                    self.vel_filtered_x = 0.0
                    self.vel_filtered_y = 0.0
                    self.deadband_locked = False
                    self.current_err_cm = 0.0

                    motor_angles = self.ik.solve(0.0, 0.0)
                    self.send_angles(motor_angles, force=True)
                else:
                    tilt_x, tilt_y = self.update_pid_2d(pos_x, pos_y, dt)
                    motor_angles = self.ik.solve(tilt_x, tilt_y)
                    self.send_angles(motor_angles, force=(tilt_x == 0.0 and tilt_y == 0.0))

                    if self.twiddle_active:
                        dist_cm = math.hypot(pos_x, pos_y) * 100.0
                        speed_cms = math.hypot(self.vel_filtered_x, self.vel_filtered_y)
                        if not self.twiddle_running_trial:
                            if dist_cm < 1.5:
                                self.twiddle_ready = True
                            if getattr(self, "twiddle_ready", True) and dist_cm > 2.2:
                                self.twiddle_running_trial = True
                                self.twiddle_ready = False
                                self.twiddle_trial_start = now
                                self.twiddle_cost = 0.0
                        else:
                            elapsed = now - self.twiddle_trial_start
                            self.twiddle_cost += elapsed * (dist_cm + 0.35 * speed_cms) * dt
                            self.twiddle_status = f"Trial {elapsed:.1f}/6.0s | Cost={self.twiddle_cost:.1f}"
                            if elapsed >= 6.0:
                                self.twiddle_running_trial = False
                                self._twiddle_step(self.twiddle_cost)
                                self.twiddle_status = f"Lift ball to reset | Next: Kp={self.Kp:.2f}, Kd={self.Kd:.2f}"

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[CONTROL] Loop error: {e}")
                break

        if self.ser:
            self.send_angles(self.ik.solve(0.0, 0.0), force=True)
            self.ser.close()

    def setup_cv_gui(self):
        win = "Stewart Platform Tracking"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self.on_mouse)
        cv2.createTrackbar("Kp (x10)", win, int(round(self.Kp * 10)), 100, lambda x: None)
        cv2.createTrackbar("Ki (x10)", win, int(round(self.Ki * 10)), 50, lambda x: None)
        cv2.createTrackbar("Kd (x10)", win, int(round(self.Kd * 10)), 50, lambda x: None)
        cv2.createTrackbar("Deadband (mm)", win, int(round(self.deadband_cm * 10)), 15, lambda x: None)
        cv2.createTrackbar("Target X", win, 50, 100, lambda x: None)
        cv2.createTrackbar("Target Y", win, 50, 100, lambda x: None)

    def poll_cv_gui(self):
        win = "Stewart Platform Tracking"
        if self.twiddle_active:
            cv2.setTrackbarPos("Kp (x10)", win, int(round(self.Kp * 10)))
            cv2.setTrackbarPos("Ki (x10)", win, 0)
            cv2.setTrackbarPos("Kd (x10)", win, int(round(self.Kd * 10)))
        else:
            self.Kp = cv2.getTrackbarPos("Kp (x10)", win) / 10.0
            self.Ki = cv2.getTrackbarPos("Ki (x10)", win) / 10.0
            self.Kd = cv2.getTrackbarPos("Kd (x10)", win) / 10.0

        self.deadband_cm = cv2.getTrackbarPos("Deadband (mm)", win) / 10.0
        raw_tx = cv2.getTrackbarPos("Target X", win)
        raw_ty = cv2.getTrackbarPos("Target Y", win)
        self.setpoint_x = (raw_tx - 50) / 1000.0
        self.setpoint_y = (raw_ty - 50) / 1000.0

    def run(self):
        print("[INFO] Controls: Drag mouse = draw path | Click / '0' = center | 't' = Twiddle auto-tune | 'q' = quit")
        self.running = True
        self.setup_cv_gui()

        cam_t = Thread(target=self.camera_thread, daemon=True)
        ctrl_t = Thread(target=self.control_thread, daemon=True)
        cam_t.start()
        ctrl_t.start()

        while self.running:
            self.poll_cv_gui()
            if self.latest_vis_frame is not None:
                frame_to_show = self.latest_vis_frame.copy()

                if len(self.drawn_points_px) > 1:
                    cv2.polylines(frame_to_show, [np.array(self.drawn_points_px, dtype=np.int32)],
                                  self.custom_closed, (255, 200, 0), 2)

                if self.custom_waypoints is not None:
                    tx_m, ty_m, _, _ = self.get_trajectory_target()
                    w = self.config['camera']['frame_width']
                    h = self.config['camera']['frame_height']
                    target_px_x = int((tx_m / self.scale_factor_x) * (w / 2.0) + (w / 2.0))
                    target_px_y = int((ty_m / self.scale_factor_y) * (h / 2.0) + (h / 2.0))
                    cv2.circle(frame_to_show, (target_px_x, target_px_y), 6, (0, 165, 255), -1)

                state_tag = "LOCKED" if self.deadband_locked else f"Err: {self.current_err_cm:.2f}cm"
                cv2.putText(frame_to_show, state_tag, (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if self.twiddle_status:
                    cv2.putText(frame_to_show, self.twiddle_status, (10, 86),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)

                cv2.imshow("Stewart Platform Tracking", frame_to_show)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('0'):
                self.clear_trajectory()
            elif key == ord('t'):
                self.toggle_twiddle()

        self.running = False
        cv2.destroyAllWindows()
        print("[INFO] Controller shut down.")


if __name__ == "__main__":
    controller = StewartController()
    controller.run()