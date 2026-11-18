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
SERVO_TRIM = [8.0, 0.0, 0.0]        # Nudge by +/- 1.0 to 4.0 deg if one corner is higher when flat
MIN_SERVO_ANGLE = 55                # Must match MIN_SAFE_ANGLE in Arduino
MAX_SERVO_ANGLE = 85                # Must match MAX_SAFE_ANGLE in Arduino

# 2. Starting PID Gains (Also adjustable live via OpenCV sliders)
DEFAULT_KP = 0.61
DEFAULT_KI = 0.0
DEFAULT_KD = 0.5
MAX_TILT_DEG = 4.0

DEADBAND_CM = 0.65                   # Ignore tiny position errors within 0.5 cm of center
MIN_ANGLE_CHANGE = 0.0              # Don't move servos unless angle changes by >= 0.3 deg

# Mouse-Drawn Trajectory Settings
TRAJ_SPEED_CM_S = 2.2               # Speed the target moves along your drawn line (cm/s)
TRAJ_RAMP_SEC = 1.5                 # Seconds to smoothly blend from current ball pos onto the path
KV_FEEDFORWARD = 0.18               # Extra tilt (deg) per (cm/s) along path tangent to overcome friction

# 3. Axis Direction Signs (Flip to -1.0 or 1.0 if an axis tilts the wrong way)
SIGN_Y_AXIS = -1.0                  # Controls front/back tilt (from camera Y)
SIGN_X_AXIS = -1.0                  # Controls left/right tilt (from camera X)

# 4. Ball Color & Size Detection (HSV)
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
            # Sub-pixel centroid (much smoother than enclosing circle center)
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
# Main Controller with Velocity-Damped PID & Mouse Trajectory
# ---------------------------------------------------------
class CoursePlateController:
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

        self.smooth_pos_x = None
        self.smooth_pos_y = None
        self.last_sent_angles = None

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
        self.stiction_boost_deg = 0.0

        # Mouse-drawn trajectory state
        self.custom_waypoints = None
        self.custom_closed = False
        self.traj_start_time = 0.0
        self.traj_start_pos = (0.0, 0.0)
        self.drawn_points_px = []
        self.is_drawing = False

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

            if not force and self.last_sent_angles is not None:
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

    def set_custom_trajectory(self, waypoints_m, closed=True):
        self.custom_closed = closed
        self.traj_start_time = time.time()
        sx = self.smooth_pos_x if self.smooth_pos_x is not None else 0.0
        sy = self.smooth_pos_y if self.smooth_pos_y is not None else 0.0
        self.traj_start_pos = (sx, sy)
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.custom_waypoints = np.array(waypoints_m, dtype=float)
        print(f"[TRAJ] Custom trajectory active ({len(waypoints_m)} pts, closed={closed})")

    def clear_trajectory(self):
        self.custom_waypoints = None
        self.drawn_points_px = []
        self.integral_x = 0.0
        self.integral_y = 0.0
        print("[TRAJ] Cleared path - returned to center hold")

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

        # Tangent velocity along path (returned in cm/s to match PID velocity units)
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

        # Smoothly blend from the ball's current position onto the drawn path
        blend = min(1.0, elapsed / TRAJ_RAMP_SEC)
        sx, sy = self.traj_start_pos
        target_x_m = (1.0 - blend) * sx + blend * raw_x
        target_y_m = (1.0 - blend) * sy + blend * raw_y
        target_vx_cms = blend * raw_vx_cms
        target_vy_cms = blend * raw_vy_cms

        return target_x_m, target_y_m, target_vx_cms, target_vy_cms

    def update_pid_2d(self, pos_x, pos_y, dt=0.033):
        # 1. Minimal position smoothing (0.85 new / 0.15 old) to prevent reversal lag
        if self.smooth_pos_x is None:
            self.smooth_pos_x, self.smooth_pos_y = pos_x, pos_y
        else:
            self.smooth_pos_x = 0.85 * pos_x + 0.15 * self.smooth_pos_x
            self.smooth_pos_y = 0.85 * pos_y + 0.15 * self.smooth_pos_y

        target_x_m, target_y_m, target_vx_cms, target_vy_cms = self.get_trajectory_target()
        is_tracking_path = self.custom_waypoints is not None

        err_x = (target_x_m - self.smooth_pos_x) * 100.0
        err_y = (target_y_m - self.smooth_pos_y) * 100.0

        # 2. Compute velocity (cm/s)
        if self.prev_pos_x is None or self.prev_pos_y is None:
            raw_vx, raw_vy = 0.0, 0.0
        else:
            raw_vx = ((self.smooth_pos_x - self.prev_pos_x) * 100.0) / dt
            raw_vy = ((self.smooth_pos_y - self.prev_pos_y) * 100.0) / dt

        self.prev_pos_x = self.smooth_pos_x
        self.prev_pos_y = self.smooth_pos_y

        self.vel_filtered_x = 0.35 * raw_vx + 0.65 * self.vel_filtered_x
        self.vel_filtered_y = 0.35 * raw_vy + 0.65 * self.vel_filtered_y

        dist_cm = math.hypot(err_x, err_y)
        speed_cms = math.hypot(self.vel_filtered_x, self.vel_filtered_y)

        # Unit direction toward target (saved before soft-deadband scaling)
        ux = (err_x / dist_cm) if dist_cm > 1e-4 else 0.0
        uy = (err_y / dist_cm) if dist_cm > 1e-4 else 0.0

        # 3. Soft deadband when parked at center (holds learned integral trim)
        if not is_tracking_path:
            if dist_cm < DEADBAND_CM and speed_cms < 1.8:
                self.stiction_boost_deg = 0.0
                tilt_x_deg = SIGN_Y_AXIS * (self.Ki * self.integral_y)
                tilt_y_deg = SIGN_X_AXIS * (self.Ki * self.integral_x)
                return (
                    math.radians(np.clip(tilt_x_deg, -MAX_TILT_DEG, MAX_TILT_DEG)),
                    math.radians(np.clip(tilt_y_deg, -MAX_TILT_DEG, MAX_TILT_DEG)),
                )

            if dist_cm < DEADBAND_CM * 2.0:
                scale = max(0.0, (dist_cm - DEADBAND_CM) / DEADBAND_CM)
                err_x *= scale
                err_y *= scale
                if speed_cms < 0.8:
                    self.vel_filtered_x = 0.0
                    self.vel_filtered_y = 0.0

        # 4. Anti-stiction boost: ramp up extra tilt if ball stops outside target, fade once rolling
        min_active_dist = 0.6 if is_tracking_path else DEADBAND_CM
        if dist_cm >= min_active_dist and speed_cms < 0.8:
            # Ramp +1.8 deg/sec while stuck (capped at +2.0 deg extra tilt)
            self.stiction_boost_deg = min(self.stiction_boost_deg + 1.8 * dt, 2.0)
        elif speed_cms > 1.2 or dist_cm < min_active_dist:
            # Drop boost quickly once static friction breaks so the ball doesn't overshoot
            self.stiction_boost_deg = max(0.0, self.stiction_boost_deg - 8.0 * dt)

        boost_x_deg = self.stiction_boost_deg * ux
        boost_y_deg = self.stiction_boost_deg * uy

        # 5. Damping
        vx, vy = self.vel_filtered_x, self.vel_filtered_y
        if is_tracking_path:
            damped_vx = self.Kd * (vx - 0.5 * target_vx_cms)
            damped_vy = self.Kd * (vy - 0.5 * target_vy_cms)
            ff_x_deg = KV_FEEDFORWARD * target_vx_cms
            ff_y_deg = KV_FEEDFORWARD * target_vy_cms
        else:
            ff_x_deg, ff_y_deg = 0.0, 0.0
            if dist_cm > DEADBAND_CM:
                rx, ry = -ux, -uy
                tx, ty = -ry, rx

                v_rad = vx * rx + vy * ry
                v_tan = vx * tx + vy * ty

                kd_rad = (self.Kd * 1.3) if v_rad > 0 else (self.Kd * 0.7)

                damped_vx = (kd_rad * v_rad * rx) + (self.Kd * v_tan * tx)
                damped_vy = (kd_rad * v_rad * ry) + (self.Kd * v_tan * ty)
            else:
                damped_vx = self.Kd * vx
                damped_vy = self.Kd * vy

        self.integral_x = np.clip(self.integral_x + err_x * dt, -15.0, 15.0)
        self.integral_y = np.clip(self.integral_y + err_y * dt, -15.0, 15.0)

        tilt_x_deg = SIGN_Y_AXIS * (self.Kp * err_y + self.Ki * self.integral_y - damped_vy + ff_y_deg + boost_y_deg)
        tilt_y_deg = SIGN_X_AXIS * (self.Kp * err_x + self.Ki * self.integral_x - damped_vx + ff_x_deg + boost_x_deg)

        tilt_x_rad = math.radians(np.clip(tilt_x_deg, -MAX_TILT_DEG, MAX_TILT_DEG))
        tilt_y_rad = math.radians(np.clip(tilt_y_deg, -MAX_TILT_DEG, MAX_TILT_DEG))
        return tilt_x_rad, tilt_y_rad

    def on_mouse(self, event, x, y, flags, param):
        # Right-click (two-finger tap on Mac) anywhere on the video -> instant reset to center
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
            # Require a real drag (at least 5 points); a click or slight twitch resets to center
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
                    self.integral_x = 0.0
                    self.integral_y = 0.0
                    self.prev_pos_x = None
                    self.prev_pos_y = None
                    self.smooth_pos_x = None
                    self.smooth_pos_y = None
                    self.vel_filtered_x = 0.0
                    self.vel_filtered_y = 0.0
                    self.stiction_boost_deg = 0.0

                    motor_angles = self.ik.solve(0.0, 0.0)
                    self.send_angles(motor_angles, force=True)
                else:
                    tilt_x, tilt_y = self.update_pid_2d(pos_x, pos_y, dt)
                    motor_angles = self.ik.solve(tilt_x, tilt_y)
                    self.send_angles(motor_angles)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[CONTROL] Loop error: {e}")
                break

        if self.ser:
            self.send_angles(self.ik.solve(0.0, 0.0), force=True)
            self.ser.close()

    def setup_cv_gui(self):
        cv2.namedWindow("Stewart Platform Tracking")
        cv2.setMouseCallback("Stewart Platform Tracking", self.on_mouse)
        cv2.createTrackbar("Kp (x10)", "Stewart Platform Tracking", int(self.Kp * 10), 100, lambda x: None)
        cv2.createTrackbar("Ki (x10)", "Stewart Platform Tracking", int(self.Ki * 10), 50, lambda x: None)
        cv2.createTrackbar("Kd (x10)", "Stewart Platform Tracking", int(self.Kd * 10), 50, lambda x: None)
        cv2.createTrackbar("Target X", "Stewart Platform Tracking", 50, 100, lambda x: None)
        cv2.createTrackbar("Target Y", "Stewart Platform Tracking", 50, 100, lambda x: None)

    def poll_cv_gui(self):
        self.Kp = cv2.getTrackbarPos("Kp (x10)", "Stewart Platform Tracking") / 10.0
        self.Ki = cv2.getTrackbarPos("Ki (x10)", "Stewart Platform Tracking") / 10.0
        self.Kd = cv2.getTrackbarPos("Kd (x10)", "Stewart Platform Tracking") / 10.0

        raw_tx = cv2.getTrackbarPos("Target X", "Stewart Platform Tracking")
        raw_ty = cv2.getTrackbarPos("Target Y", "Stewart Platform Tracking")
        self.setpoint_x = (raw_tx - 50) / 1000.0
        self.setpoint_y = (raw_ty - 50) / 1000.0

    def run(self):
        print("[INFO] Starting Stewart Platform Controller (Drag mouse to draw path, '0' to clear, 'q' to exit)...")
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

                cv2.imshow("Stewart Platform Tracking", frame_to_show)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('0'):
                self.clear_trajectory()

        self.running = False
        cv2.destroyAllWindows()
        print("[INFO] Controller shut down.")


if __name__ == "__main__":
    controller = CoursePlateController()
    controller.run()