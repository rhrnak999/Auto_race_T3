#!/usr/bin/env python3
#
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Leon Jung, Gilbert, Ashe Kim, Hyungyu Kim, ChanHyeong Lee
import sys
import os

# 설치된 패키지 경로 추가
src_path = 'src/turtlebot3_autorace/turtlebot3_autorace_mission/turtlebot3_autorace_mission/'
if src_path not in sys.path:
    sys.path.append(src_path)

# 디버깅용: sys.path 출력
print("Python path:", sys.path)

try:
    from traffic_light import ControlTraffic
    print("Successfully imported ControlTraffic")
except ImportError as e:
    print("Failed to import ControlTraffic:", e)

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, UInt8, Float64
from sensor_msgs.msg import Imu
from tf_transformations import euler_from_quaternion  # quaternion → roll, pitch, yaw 변환
from nav_msgs.msg import Odometry
import time

class ControlLane(Node):
    def __init__(self):
        super().__init__('control_lane')
        self.ctt = ControlTraffic()
        # 신호등 상태 구독자
        self.sub_traffic_light = self.create_subscription(
            UInt8,
            '/detect/traffic_light',
            self.callback_traffic_light,
            1
        )

        # 신뢰도 정보 구독자들
        self.sub_red_reliability = self.create_subscription(
            UInt8,
            '/detect/red_light_reliability',
            self.callback_red_reliability,
            1
        )

        self.sub_yellow_reliability = self.create_subscription(
            UInt8,
            '/detect/yellow_light_reliability',
            self.callback_yellow_reliability,
            1
        )
        self.sub_green_reliability = self.create_subscription(
            UInt8,
            '/detect/green_light_reliability',
            self.callback_green_reliability,
            1
        )

        self.sub_lane = self.create_subscription(
            Float64,
            '/control/lane',
            self.callback_follow_lane,
            1
        )

        # self.sub_max_vel = self.create_subscription(
        #     Float64,
        #     '/control/max_vel',
        #     self.callback_get_max_vel,
        #     1
        # )
        self.sub_avoid_cmd = self.create_subscription(
            Twist,
            '/avoid_control',
            self.callback_avoid_cmd,
            1
        )
        self.sub_avoid_active = self.create_subscription(
            Bool,
            '/avoid_active',
            self.callback_avoid_active,
            1
        )

        self.pub_cmd_vel = self.create_publisher(
            Twist,
            '/control/cmd_vel',
            1
        )
        self.sub_ctrl_vel = self.create_subscription(
            Twist,
            '/ctrl_vel',
            self.callback_ctrl_vel,
            1
        )

        self.sub_sign = self.create_subscription(
            UInt8,
            '/detect/traffic_sign/parking',
            self.callback_detect_parking,
            10
        )

        self.sign_detected = False  # 중복 방지용

        #imu값 받아오기
        self.sub_imu = self.create_subscription(
            Imu,
            '/imu',
            self.callback_imu,
            10
        )

        self.current_yaw = 0.0
        self.yaw_triggered = False  # 중복 방지

        self.current_position = (0.0, 0.0)

        self.sub_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.callback_odom,
            10
        )
        
        self.parking_mode = False         # 파킹 시작 여부
        self.parking_completed = False    # 파킹 완료 여부
        self.TARGET_POS = (0.45, 1.75)
        self.TOLERANCE_DIST = 0.05
        self.TARGET_YAW = 3.14
        self.TOLERANCE_YAW = 0.1
        self.last_log_time = time.time()
        
        # PD control related variables
        self.last_error = 0
        self.CTRL_VEL = Twist()

        # Avoidance mode related variables
        self.avoid_active = False
        self.avoid_twist = Twist()

    def callback_ctrl_vel(self,ctrl_vel_msg):
        self.CTRL_VEL = ctrl_vel_msg
    # def callback_get_max_vel(self, max_vel_msg):
    #     self.CTRL_VEL = max_vel_msg.data

    def callback_red_reliability(self, msg):
        """빨간불 신뢰도 업데이트"""
        self.ctt.red_reliability = msg.data

    def callback_yellow_reliability(self, msg):
        """노란불 신뢰도 업데이트"""
        self.ctt.yellow_reliability = msg.data

    def callback_green_reliability(self, msg):
        """초록불 신뢰도 업데이트"""
        self.ctt.green_reliability = msg.data

    def callback_traffic_light(self, traffic_light_msg):
        """신뢰도를 고려한 신호등 상태 업데이트"""
        raw_state = traffic_light_msg.data
        # 신뢰도 체크
        is_reliable = self.ctt.check_state_reliability(raw_state)
        
        if is_reliable:
            # 신뢰도가 충분하면 정상 처리
            processed_state = self.ctt.process_traffic_state_with_persistence(raw_state)
            
            if processed_state != self.ctt.current_traffic_state:
                self.ctt.current_traffic_state = processed_state
                self.ctt.log_traffic_state_change()
                self.ctt.reliability_override_count = 0
        else:
            # 신뢰도가 낮으면 이전 상태 유지
            self.ctt.reliability_override_count += 1
            
            # 주기적으로 경고 로그 (빈도 감소)
            if self.ctt.reliability_override_count % 20 == 1:  # 10 → 20 (덜 자주 출력)
                self.get_logger().warn(f'Low reliability: {self.ctt.get_state_name(raw_state)} - maintaining current state')

        #####재권 추가 시작(주차 사인 감지됐을 때 주차 동작)########################3
    def callback_odom(self, msg):
        pos = msg.pose.pose.position
        self.current_position = (pos.x, pos.y)
    def sol_parking(self):
        tx, ty = self.TARGET_POS
        dist = ((self.current_position[0] - tx)**2 + (self.current_position[1] - ty)**2)**0.5
        current_time = time.time()
        if current_time - self.last_log_time >= 2.0:
            self.get_logger().info(f"[주차 접근중] 위치: x={self.current_position[0]:.2f}, y={self.current_position[1]:.2f}, 거리={dist:.3f}")
            self.last_log_time = current_time

        if dist > self.TOLERANCE_DIST:
            # 아직 목표 위치 도달 전 → 직진 명령
            twist = Twist()
            twist.linear.x = self.CTRL_VEL.linear.x
            twist.angular.z = 0.0
            self.pub_cmd_vel.publish(twist)
        else:
            # 목표 위치 도달 → yaw 확인
            yaw = self.current_yaw
            if abs(abs(yaw) - self.TARGET_YAW) < self.TOLERANCE_YAW:

                park1_twist = Twist()
                park1_twist.linear.x = 0.0
                park1_twist.angular.z = 0.0
                self.pub_cmd_vel.publish(park1_twist)
                time.sleep(3.0)

                self.get_logger().info(f"[주차] 목표 위치 및 방향 만족 → 회전 시작")
                park2_twist = Twist()
                park2_twist.linear.x = -0.02
                park2_twist.angular.z = -0.93
                self.pub_cmd_vel.publish(park2_twist)
                time.sleep(2.0)

                self.get_logger().info('주차... 후진 중')
                park3_twist = Twist()
                park3_twist.linear.x = -0.1
                park3_twist.angular.z = 0.0
                self.pub_cmd_vel.publish(park3_twist)
                time.sleep(9.0)

                stop_twist = Twist()
                stop_twist.linear.x = 0.0
                stop_twist.angular.z = 0.0
                self.pub_cmd_vel.publish(stop_twist)  # 정지
                time.sleep(3.0)
                self.get_logger().info("주차 완료")

                start_twist = Twist()
                start_twist.linear.x = 0.1
                start_twist.angular.z = 0.0
                self.pub_cmd_vel.publish(start_twist)  # 앞으로 전진
                time.sleep(7.0)
                self.get_logger().info("주행 시작")

                start2_twist = Twist()
                start2_twist.linear.x = 0.1
                start2_twist.angular.z = 0.5
                self.pub_cmd_vel.publish(start2_twist)  # 앞으로 전진
                time.sleep(3.0)
                self.get_logger().info("좌회전 완료")

                self.parking_completed = True
                self.parking_mode = False
            else:
                self.get_logger().info(f"[대기] yaw {round(yaw, 2)} → 3.14 도달 대기중")

    def callback_imu(self, msg):
        # quaternion → euler 변환
        q = msg.orientation
        quaternion = (q.x, q.y, q.z, q.w)
        (roll, pitch, yaw) = euler_from_quaternion(quaternion)

        self.current_yaw = yaw  # rad 단위 (-π ~ π)

        # 로그
        # self.get_logger().info(f'현재 yaw(rad): {round(yaw, 2)}')

    def callback_detect_parking(self, msg):
        if msg.data == 1 and not self.parking_mode:
            self.get_logger().info("Parking sign detected. 주차 모드 진입")
            self.parking_mode = True
            self.parking_completed = False
            self.sign_detected = True
    def callback_follow_lane(self, desired_center):
        """
        Receive lane center data to generate lane following control commands.

        If avoidance mode is enabled, lane following control is ignored.
        """
        twist = Twist()
        if self.avoid_active:
            return
        elif self.parking_mode and not self.parking_completed:
            self.sol_parking()
            return
        elif not self.ctt.current_traffic_state==0:
            twist.linear.x = self.ctt.apply_traffic_control(self.CTRL_VEL.linear.x)
            twist.angular.z = 0.0
            self.pub_cmd_vel.publish(twist)
            return
        center = desired_center.data
        error = center - 500

        Kp = 0.0025
        Kd = 0.007

        angular_z = Kp * error + Kd * (error - self.last_error)
        self.last_error = error

        twist.linear.x = min(self.CTRL_VEL.linear.x * (max(1 - abs(error) / 500, 0) ** 2.2), 0.22)  #추후 제한치 변경 예정 실제에선 0.22로 하면 고장남
        twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)
        self.pub_cmd_vel.publish(twist)

    def callback_avoid_cmd(self, twist_msg):
        self.avoid_twist = twist_msg

        if self.avoid_active:
            self.pub_cmd_vel.publish(self.avoid_twist)

    def callback_avoid_active(self, bool_msg):
        self.avoid_active = bool_msg.data
        if self.avoid_active:
            self.get_logger().info('Avoidance mode activated.')
        else:
            self.get_logger().info('Avoidance mode deactivated. Returning to lane following.')

    def shut_down(self):
        self.get_logger().info('Shutting down. cmd_vel will be 0')
        twist = Twist()
        self.pub_cmd_vel.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ControlLane()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shut_down()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
