#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QImage, QFont, QPixmap
from sensor_msgs.msg import Image, Imu, LaserScan, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import cv2
from cv_bridge import CvBridge
import numpy as np
try:
    from tf_transformations import euler_from_quaternion
except ImportError as e:
    print(f"Error importing tf_transformations: {e}")
    sys.exit(1)

# Load static map image
MAP_PATH = "/home/minsuje/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_autorace_2020/course/materials/textures/course.png"
map_image = QImage(MAP_PATH)
if map_image.isNull():
    print(f"Error: Could not load map image from {MAP_PATH}. Check file path and permissions.")
    sys.exit(1)

# 속도 조절 PyQt5
class KnobWidget(QLabel):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.label = label
        self.value = 0.0  # 0~792 m/h
        self.angle = 0.0
        self.dragging = False
        self.setFixedSize(120, 120)
        self.setStyleSheet("background: #FFFFFF; border: 2px solid #000000;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#000000"), 4))
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.drawEllipse(10, 10, 100, 100)
        painter.setPen(QPen(QColor("#2196F3"), 2))
        painter.setBrush(QBrush(QColor("#2196F3")))
        center = QPointF(60, 60)
        radius = 35
        rad = np.deg2rad(self.angle)
        knob_x = center.x() + radius * np.cos(rad)
        knob_y = center.y() - radius * np.sin(rad)
        painter.drawEllipse(QPointF(knob_x, knob_y), 12, 12)
        painter.setFont(QFont("Roboto", 14, QFont.Bold))
        painter.setPen(QPen(QColor("#000000"), 2))
        speed_mph = self.value
        text = f"{speed_mph:.0f} m/h"
        painter.drawText(QRectF(10, 80, 100, 30), Qt.AlignCenter, text)
        painter.end()
        print(f"Knob {self.label} value: {text}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.update_angle(event.pos())

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.update_angle(event.pos())

    def mouseReleaseEvent(self, event):
        self.dragging = False

    def update_angle(self, pos):
        center = QPointF(60, 60)
        vec = pos - center
        angle = np.arctan2(-vec.y(), vec.x()) * 180 / np.pi
        self.angle = (angle + 360) % 360
        self.value = (self.angle / 360.0) * 792  # 0~792 m/h
        self.update()

# 현재 실제 속도 표시
class OdomSpeedWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.linear_vel = 0.0
        self.setFixedSize(200, 200)
        self.setStyleSheet("background: #FFFFFF; border: 4px solid #000000;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(20, 20, 160, 160)
        painter.setPen(QPen(QColor("#000000"), 6))
        painter.drawArc(rect, 0, 360 * 16)
        max_speed = 0.22
        angle = int((self.linear_vel / max_speed) * 300 * 16) if self.linear_vel >= 0 else 0
        painter.setPen(QPen(QColor("#2196F3"), 4, Qt.SolidLine))
        if angle > 180 * 16:
            painter.setPen(QPen(QColor("#FF0000"), 4, Qt.SolidLine))
        painter.drawArc(rect, 30 * 16, angle)
        painter.setFont(QFont("Roboto", 16, QFont.Bold))
        painter.setPen(QPen(QColor("#000000"), 3))
        painter.drawText(QRectF(20, 90, 160, 40), Qt.AlignCenter, f"Odom: {self.linear_vel * 3600:.0f} m/h")
        painter.end()
        print(f"Odom Speed: {self.linear_vel * 3600:.0f} m/h")

class TurtleBotDashboard(Node, QMainWindow):
    def __init__(self):
        super().__init__('turtlebot_dashboard')
        QMainWindow.__init__(self)
        self.setWindowTitle("TurtleBot3 Dashboard")
        self.setStyleSheet("""
            QMainWindow { background-color: #FFFFFF; }
            QLabel { color: #000000; font-family: Roboto; font-size: 14px; }
            QPushButton {
                background-color: #2196F3; color: white;
                border-radius: 5px; padding: 8px;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        self.setGeometry(0, 0, 1600, 1200)
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/ctrl_vel', 10)
        self.vel_pub = self.create_publisher(Twist, '/vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_color', self.image_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_avoid_active = self.create_subscription(Bool, '/avoid_active', self.callback_avoid_active, 10)
        self.bridge = CvBridge()

        self.avoid_active=False
        self.linear_vel = 0.0
        self.odom_linear_vel = 0.0
        self.cmd_linear = 0.0
        self.linear_acc = 0.0
        self.yaw = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.scan_data = None
        self.last_odom_time = self.get_clock().now()
        self.last_odom_vel = 0.0
        
        self.linear_integral = 0.0
        self.linear_prev_error = 0.0
        self.prev_derivative_linear = 0.0
        self.cmd_buffer = {'linear': []}
        self.cmd_buffer_size = 1
        self.last_ctrl_time = self.get_clock().now()
        self.last_integral_reset_time = self.get_clock().now()  # 적분항 리셋 시간
        
        self.setup_ui()
        self.apply_cmd_vel()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100)
        self.ctrl_timer = QTimer(self)
        self.ctrl_timer.timeout.connect(self.apply_cmd_vel)
        self.ctrl_timer.start(100)
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)
        
        self.odom_speed_widget = OdomSpeedWidget()
        layout.addWidget(self.odom_speed_widget, 0, 0)
        
        self.linear_knob = KnobWidget("Linear")
        knob_widget = QWidget()
        knob_layout = QGridLayout(knob_widget)
        knob_layout.addWidget(self.linear_knob, 0, 0)
        layout.addWidget(knob_widget, 1, 0)
        
        button_widget = QWidget()
        button_layout = QGridLayout(button_widget)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_robot)
        button_layout.addWidget(self.stop_btn, 0, 0)
        layout.addWidget(button_widget, 2, 0)
        
        self.map_label = QLabel()
        self.map_label.setPixmap(QPixmap.fromImage(map_image).scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.map_label.setFixedSize(800, 800)
        layout.addWidget(self.map_label, 0, 1, 3, 1)
        
        self.pos_label = QLabel("X: 0.0, Y: 0.0")
        self.pos_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.pos_label, 3, 1)
        
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(800, 800)
        self.camera_label.setStyleSheet("border: 2px solid #424242;")
        layout.addWidget(self.camera_label, 0, 2, 4, 1)
        
    def odom_callback(self, msg):
        """오도메트리 처리: 노이즈 필터링 강화"""
        try:
            current_time = self.get_clock().now()
            raw_vel = max(0.0, msg.twist.twist.linear.x)  # 음수 속도 방지
            # 최초에 너무 빠른 속도 적용 때문에 진동이 심해서 노이즈 158~558의 alpha로 노이즈 제거 목적으로 했음. 다만 실제 상황과 다르다는 점이에초에 건드릴 일이 아니기에 노이즈 필터는 두고 그대로 사용중.
            alpha = 0.1  # 강한 필터링 (노이즈 158~558 m/h 완화)Exponential Moving Average (EMA) 지수 이동 필터(저역통과 필터)
            self.odom_linear_vel = alpha * raw_vel + (1 - alpha) * self.odom_linear_vel
            self.pos_x = msg.pose.pose.position.x
            self.pos_y = msg.pose.pose.position.y
            orientation = msg.pose.pose.orientation
            _, _, self.yaw = euler_from_quaternion([
                orientation.x, orientation.y, orientation.z, orientation.w])
            dt = (current_time - self.last_odom_time).nanoseconds / 1e9
            if dt > 0 and self.last_odom_vel != 0:
                self.linear_acc = (self.odom_linear_vel - self.last_odom_vel) / dt
            self.last_odom_vel = self.odom_linear_vel
            self.last_odom_time = current_time
            self.get_logger().info(f"Odom - Pos: ({self.pos_x:.2f}, {self.pos_y:.2f}), Vel: {self.odom_linear_vel * 3600:.0f} m/h")
        except Exception as e:
            self.get_logger().warn(f"Odom callback error: {e}")
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = cv2.resize(cv_image, (800, 800))
            height, width, channel = cv_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            q_image = q_image.rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio))
        except Exception as e:
            self.get_logger().warn(f"Image callback error: {e}")
        
    def scan_callback(self, msg):
        try:
            self.scan_data = msg
        except Exception as e:
            self.get_logger().warn(f"Scan callback error: {e}")
        
    def update_ui(self):
        self.odom_speed_widget.linear_vel = self.odom_linear_vel
        self.odom_speed_widget.update()
        
        self.pos_label.setText(f"X: {self.pos_x:.2f}, Y: {self.pos_y:.2f}")
        
        pixmap = QPixmap.fromImage(map_image).scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        map_scale = 200  # 1m = 200px
        map_center_x = 400
        map_center_y = 400
        robot_x = map_center_x - self.pos_x * map_scale
        robot_y = self.pos_y * map_scale + map_center_y
        robot_x = max(0, min(800, robot_x))
        robot_y = max(0, min(800, robot_y))
        self.get_logger().info(f"Robot Pos (pix): ({robot_x:.2f}, {robot_y:.2f})")
        
        display_yaw = (self.yaw + np.pi) % (2 * np.pi)
        painter.setPen(QPen(QColor("blue"), 5))
        painter.setBrush(QBrush(QColor("blue")))
        painter.save()
        painter.translate(robot_x, robot_y)
        painter.rotate(-np.degrees(display_yaw))
        painter.drawEllipse(QPointF(0, 0), 5, 5)
        painter.drawLine(0, 0, 20, 0)
        painter.restore()
        
        if self.scan_data is not None:
            angles = np.linspace(self.scan_data.angle_min, self.scan_data.angle_max, len(self.scan_data.ranges))
            for i, r in enumerate(self.scan_data.ranges):
                if r < self.scan_data.range_max:
                    scan_angle = angles[i] + display_yaw
                    scan_x = robot_x + r * map_scale * np.cos(scan_angle)
                    scan_y = robot_y - r * map_scale * np.sin(scan_angle)
                    scan_x = max(0, min(800, scan_x))
                    scan_y = max(0, min(800, scan_y))
                    painter.setPen(QPen(QColor("red"), 2))
                    painter.drawPoint(QPointF(scan_x, scan_y))
        
        painter.end()
        self.map_label.setPixmap(pixmap)
    
    def callback_avoid_active(self, bool_msg):
        self.avoid_active = bool_msg.data
        if self.avoid_active:
            self.get_logger().info('Avoidance mode activated.')
        else:
            self.get_logger().info('Avoidance mode deactivated. Returning to lane following.')

    def windup_control(self, error, prev_error, integral, dt, kp=1.2, ki=0.005, kd=0.01, output_limit=0.05):
        """PID 제어: 미분항ema, 적분항 축적 완화"""
        proportional = kp * error
        derivative = kd * (error - prev_error) / dt
        alpha = 0.2
        prev_derivative = getattr(self, 'prev_derivative_linear', 0.0)
        filtered_derivative = alpha * derivative + (1 - alpha) * prev_derivative
        setattr(self, 'prev_derivative_linear', filtered_derivative)
        delta_cmd = proportional + integral + filtered_derivative
        max_accel = 0.5  # 증가 가속도
        max_decel = 1.0  # 감속도 (응답성 강화)
        max_delta = max_decel * dt if error < 0 else max_accel * dt
        if abs(delta_cmd) < output_limit:
            integral += ki * error * dt
            # 적분항 제한 강화
            integral = max(-output_limit, min(output_limit, integral))
        if delta_cmd > max_delta:
            integral -= (delta_cmd - max_delta) / kp
            delta_cmd = max_delta
        elif delta_cmd < -max_delta:
            integral -= (delta_cmd + max_delta) / kp
            delta_cmd = -max_delta
        if self.cmd_linear + delta_cmd < 0:
            delta_cmd = -self.cmd_linear
            integral = 0.0
        return delta_cmd, error, integral

    def apply_cmd_vel(self):
        """속도 변화량 PID, 1초마다 적분항 초기화"""
        if(not self.avoid_active):
            current_time = self.get_clock().now()
            dt = max(0.05, min(0.2, (current_time - self.last_ctrl_time).nanoseconds / 1e9))
            self.last_ctrl_time = current_time

            # 1초마다 적분항 초기화
            if (current_time - self.last_integral_reset_time).nanoseconds / 1e9 >= 1.0:
                self.linear_integral = 0.0
                self.last_integral_reset_time = current_time
                self.get_logger().info("Linear integral reset to 0")

            target_linear = (self.linear_knob.value / 792) * 0.22  # 792 m/h = 0.22 m/s
            test_vel = Twist()
            test_vel.linear.x = target_linear
            linear_error = 0.0  # 초기화
            delta_cmd = 0.0
            if abs(target_linear) < 1e-4:  # 목표 0일 때 정지
                self.cmd_linear = 0.0
                self.linear_integral = 0.0
                self.linear_prev_error = 0.0
            else:
                linear_error = target_linear - self.odom_linear_vel
                delta_cmd, self.linear_prev_error, self.linear_integral = self.windup_control(
                    linear_error, self.linear_prev_error, self.linear_integral, dt,
                    kp=1.2, ki=0.05, kd=0.1, output_limit=0.01
                )
                self.cmd_linear += delta_cmd

            self.cmd_linear = max(0.0, min(0.22, self.cmd_linear))

            self.cmd_buffer['linear'].append(self.cmd_linear)
            if len(self.cmd_buffer['linear']) > self.cmd_buffer_size:
                self.cmd_buffer['linear'].pop(0)

            twist = Twist()
            twist.linear.x = sum(self.cmd_buffer['linear']) / len(self.cmd_buffer['linear'])
            try:
                self.vel_pub.publish(test_vel)
                self.cmd_vel_pub.publish(twist)
                self.get_logger().info(
                    f"Applied - Knob: {self.linear_knob.value:.0f} m/h, Target: {target_linear * 3600:.0f} m/h, "
                    f"PID Out: {delta_cmd * 3600:.0f} m/h, Cmd: {twist.linear.x * 3600:.0f} m/h, "
                    f"Error: {linear_error:.3f}")
            except Exception as e:
                self.get_logger().warn(f"Cmd_vel publish error: {e}")
            else:
                twist = Twist()
                twist.linear.x = (self.linear_knob.value / 792) * 0.22
                self.cmd_vel_pub.publish(twist)
                self.linear_integral = 0.0
                self.last_integral_reset_time = current_time
                self.get_logger().info("Linear integral reset to 0")
        
    def stop_robot(self):
        """로봇 정지 및 PID 상태 초기화"""
        self.cmd_linear = 0.0
        twist = Twist()
        try:
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info("Robot stopped")
        except Exception as e:
            self.get_logger().warn(f"Stop robot error: {e}")
        self.linear_knob.value = 0.0
        self.linear_knob.angle = 0.0
        self.linear_knob.update()
        self.linear_integral = 0.0
        self.linear_prev_error = 0.0
        self.prev_derivative_linear = 0.0
        self.last_integral_reset_time = self.get_clock().now()

if __name__ == '__main__':
    try:
        rclpy.init()
        app = QApplication(sys.argv)
        node = TurtleBotDashboard()
        node.show()
        timer = QTimer()
        timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0))
        timer.start(10)
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()