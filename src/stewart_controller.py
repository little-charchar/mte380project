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
USB_PORT = "/dev/cu.usbmodem11401"   # Run `ls /dev/cu.usb*` in terminal to check
BAUD_RATE = 115200
CAMERA_INDEX = 0

# 1. Neutral & Leveling Trim
PHYSICAL_NEUTRAL = 70.0             # Must match NEUTRAL_ANGLE in Arduino
SERVO_TRIM = [8.0, 0.0, 0.0]        # Nudge by +/- 1.0 to 4.0 deg if one corner is higher when flat
MIN_SERVO_ANGLE = 55                # Must match MIN_SAFE_ANGLE in Arduino
MAX_SERVO_ANGLE = 85                # Must match MAX_SAFE_ANGLE in Arduino

# 2. Starting PID Gains (Also adjustable live via OpenCV sliders)
DEFAULT_KP = 0.6
DEFAULT_KI = 0.1
DEFAULT_KD = 0.6
KD_TANGENTIAL = 0.6
MAX_TILT_DEG = 4.0

DEADBAND_CM = 0.4                  # Ignore tiny position errors within 0.5 mm of center
MIN_ANGLE_CHANGE = 0             # Don't move servos unless angle changes by >= 0.3 deg

# Mouse-Drawn Trajectory Settings
TRAJ_SPEED_CM_S = 2.2       # Speed the target moves along your drawn line (cm/s)
TRAJ_RAMP_SEC = 1.5         # Seconds to smoothly blend from current ball pos onto the path
KV_FEEDFORWARD = 0.18       # Extra tilt per (cm/s) along path tangent to overcome friction

# 3. Axis Direction Signs (Flip to -1.0 or 1.0 if an axis tilts the wrong way)
SIGN_Y_AXIS = -1.0                   # Controls front/back tilt (from camera Y)
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
# Main Controller with Velocity-Damped PID
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

    def update_pid_2d(self, pos_x, pos_y, dt=0.033):
        # 1. Minimal position smoothing (0.85 new / 0.15 old) to prevent reversal lag
        if self.smooth_pos_x is None:
            self.smooth_pos_x, self.smooth_pos_y = pos_x, pos_y
        else:
            self.smooth_pos_x = 0.85 * pos_x + 0.15 * self.smooth_pos_x
            self.smooth_pos_y = 0.85 * pos_y + 0.15 * self.smooth_pos_y

        err_x = (self.setpoint_x - self.smooth_pos_x) * 100.0
        err_y = (self.setpoint_y - self.smooth_pos_y) * 100.0

        # 2. Compute velocity (cm/s)
        if self.prev_pos_x is None or self.prev_pos_y is None:
            raw_vx, raw_vy = 0.0, 0.0
        else:
            raw_vx = ((self.smooth_pos_x - self.prev_pos_x) * 100.0) / dt
            raw_vy = ((self.smooth_pos_y - self.prev_pos_y) * 100.0) / dt

        self.prev_pos_x = self.smooth_pos_x
        self.prev_pos_y = self.smooth_pos_y

        # Snappy velocity filter (0.8 new / 0.2 old) so direction reversals register immediately
        self.vel_filtered_x = 0.8 * raw_vx + 0.2 * self.vel_filtered_x
        self.vel_filtered_y = 0.8 * raw_vy + 0.2 * self.vel_filtered_y

        # 3. Soft deadband & camera-noise gate near center
        dist_cm = math.hypot(err_x, err_y)
        speed_cms = math.hypot(self.vel_filtered_x, self.vel_filtered_y)
        if dist_cm < DEADBAND_CM and speed_cms < 0.8:
            self.integral_x = 0.0
            self.integral_y = 0.0
            return 0.0, 0.0

        # Smoothly ramp corrections outside DEADBAND_CM so crossing the edge doesn't kick the ball
        if dist_cm < DEADBAND_CM * 2.0:
            scale = max(0.0, (dist_cm - DEADBAND_CM) / DEADBAND_CM)
            err_x *= scale
            err_y *= scale
            if speed_cms < 0.8:
                self.vel_filtered_x = 0.0
                self.vel_filtered_y = 0.0

         # 4. Velocity damping (same in every direction)
        damped_vx = self.Kd * self.vel_filtered_x
        damped_vy = self.Kd * self.vel_filtered_y


        self.integral_x = np.clip(self.integral_x + err_x * dt, -15.0, 15.0)
        self.integral_y = np.clip(self.integral_y + err_y * dt, -15.0, 15.0)

        tilt_x_deg = SIGN_Y_AXIS * (self.Kp * err_y + self.Ki * self.integral_y - damped_vy)
        tilt_y_deg = SIGN_X_AXIS * (self.Kp * err_x + self.Ki * self.integral_x - damped_vx)

        tilt_x_rad = math.radians(np.clip(tilt_x_deg, -MAX_TILT_DEG, MAX_TILT_DEG))
        tilt_y_rad = math.radians(np.clip(tilt_y_deg, -MAX_TILT_DEG, MAX_TILT_DEG))
        return tilt_x_rad, tilt_y_rad

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
                    # 1. Wipe all PID and smoothing memory while no ball is on the plate
                    self.integral_x = 0.0
                    self.integral_y = 0.0
                    self.prev_pos_x = None
                    self.prev_pos_y = None
                    self.smooth_pos_x = None
                    self.smooth_pos_y = None
                    self.vel_filtered_x = 0.0
                    self.vel_filtered_y = 0.0

                    # 2. Force the servos all the way back to level (bypassing MIN_ANGLE_CHANGE)
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
        print("[INFO] Starting Stewart Platform Controller (Press 'q' in video window to exit)...")
        self.running = True
        self.setup_cv_gui()

        cam_t = Thread(target=self.camera_thread, daemon=True)
        ctrl_t = Thread(target=self.control_thread, daemon=True)
        cam_t.start()
        ctrl_t.start()

        while self.running:
            self.poll_cv_gui()
            if self.latest_vis_frame is not None:
                cv2.imshow("Stewart Platform Tracking", self.latest_vis_frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        self.running = False
        cv2.destroyAllWindows()
        print("[INFO] Controller shut down.")


if __name__ == "__main__":
    controller = CoursePlateController()
    controller.run()