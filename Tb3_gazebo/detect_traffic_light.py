#ligth jjj
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
# Author: Leon Jung, Gilbert, Ashe Kim, ChanHyeong Lee

import time

import cv2
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import numpy as np
from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8


class DetectTrafficLight(Node):

    def __init__(self):
        super().__init__('detect_traffic_light')
        parameter_descriptor_hue = ParameterDescriptor(
            integer_range=[IntegerRange(from_value=0, to_value=179, step=1)],
            description='Hue Value (0~179)'
        )
        parameter_descriptor_saturation_lightness = ParameterDescriptor(
            integer_range=[IntegerRange(from_value=0, to_value=255, step=1)],
            description='Saturation/Lightness Value (0~255)'
        )

        self.declare_parameter(
            'red.hue_l', 0, parameter_descriptor_hue)
        self.declare_parameter(
            'red.hue_h', 179, parameter_descriptor_hue)
        self.declare_parameter(
            'red.saturation_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'red.saturation_h', 255, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'red.lightness_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'red.lightness_h', 255, parameter_descriptor_saturation_lightness)

        self.declare_parameter(
            'yellow.hue_l', 0, parameter_descriptor_hue)
        self.declare_parameter(
            'yellow.hue_h', 179, parameter_descriptor_hue)
        self.declare_parameter(
            'yellow.saturation_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'yellow.saturation_h', 255, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'yellow.lightness_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'yellow.lightness_h', 255, parameter_descriptor_saturation_lightness)

# 초록색 (파란색과 확실히 구분되도록 설정)
        self.declare_parameter('green.hue_l', 35, parameter_descriptor_hue)    # 파란색(94~100)과 확실히 구분
        self.declare_parameter('green.hue_h', 85, parameter_descriptor_hue)    # 파란색 피하기 (80→85로 확장)
        self.declare_parameter('green.saturation_l', 80, parameter_descriptor_saturation_lightness)  # 채도 조건
        self.declare_parameter('green.saturation_h', 255, parameter_descriptor_saturation_lightness)
        self.declare_parameter('green.lightness_l', 80, parameter_descriptor_saturation_lightness)   # 밝기 조건
        self.declare_parameter('green.lightness_h', 255, parameter_descriptor_saturation_lightness)

        self.declare_parameter('is_calibration_mode', False)

        # 파란색 표지판 필터링을 위한 HSV 상수 추가
        self.BLUE_SIGN_H_MIN = 94    # 97-3
        self.BLUE_SIGN_H_MAX = 100   # 97+3  
        self.BLUE_SIGN_S_MIN = 168   # 183-15
        self.BLUE_SIGN_V_MIN = 200   # 215-15

        self.hue_red_l = self.get_parameter(
            'red.hue_l').get_parameter_value().integer_value
        self.hue_red_h = self.get_parameter(
            'red.hue_h').get_parameter_value().integer_value
        self.saturation_red_l = self.get_parameter(
            'red.saturation_l').get_parameter_value().integer_value
        self.saturation_red_h = self.get_parameter(
            'red.saturation_h').get_parameter_value().integer_value
        self.lightness_red_l = self.get_parameter(
            'red.lightness_l').get_parameter_value().integer_value
        self.lightness_red_h = self.get_parameter(
            'red.lightness_h').get_parameter_value().integer_value

        self.hue_yellow_l = self.get_parameter(
            'yellow.hue_l').get_parameter_value().integer_value
        self.hue_yellow_h = self.get_parameter(
            'yellow.hue_h').get_parameter_value().integer_value
        self.saturation_yellow_l = self.get_parameter(
            'yellow.saturation_l').get_parameter_value().integer_value
        self.saturation_yellow_h = self.get_parameter(
            'yellow.saturation_h').get_parameter_value().integer_value
        self.lightness_yellow_l = self.get_parameter(
            'yellow.lightness_l').get_parameter_value().integer_value
        self.lightness_yellow_h = self.get_parameter(
            'yellow.lightness_h').get_parameter_value().integer_value

        self.hue_green_l = self.get_parameter(
            'green.hue_l').get_parameter_value().integer_value
        self.hue_green_h = self.get_parameter(
            'green.hue_h').get_parameter_value().integer_value
        self.saturation_green_l = self.get_parameter(
            'green.saturation_l').get_parameter_value().integer_value
        self.saturation_green_h = self.get_parameter(
            'green.saturation_h').get_parameter_value().integer_value
        self.lightness_green_l = self.get_parameter(
            'green.lightness_l').get_parameter_value().integer_value
        self.lightness_green_h = self.get_parameter(
            'green.lightness_h').get_parameter_value().integer_value

        self.is_calibration_mode = self.get_parameter(
            'is_calibration_mode').get_parameter_value().bool_value
        if self.is_calibration_mode:
            self.add_on_set_parameters_callback(self.get_detect_traffic_light_param)

        self.sub_image_type = 'raw'
        self.pub_image_type = 'compressed'

        self.counter = 1

        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage, '/detect/image_input/compressed', self.get_image, 1)
        else:
            self.sub_image_original = self.create_subscription(
                Image, '/detect/image_input', self.get_image, 1)

        if self.pub_image_type == 'compressed':
            self.pub_image_traffic_light = self.create_publisher(
                CompressedImage, '/detect/image_output/compressed', 1)
        else:
            self.pub_image_traffic_light = self.create_publisher(
                Image, '/detect/image_output', 1)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_red_light = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub1/compressed', 1)
                self.pub_image_yellow_light = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub2/compressed', 1)
                self.pub_image_green_light = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub3/compressed', 1)
                # 파란색 필터링 결과를 위한 퍼블리셔 추가
                self.pub_image_blue_filtered = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub4/compressed', 1)
            else:
                self.pub_image_red_light = self.create_publisher(
                    Image, '/detect/image_output_sub1', 1)
                self.pub_image_yellow_light = self.create_publisher(
                    Image, '/detect/image_output_sub2', 1)
                self.pub_image_green_light = self.create_publisher(
                    Image, '/detect/image_output_sub3', 1)
                # 파란색 필터링 결과를 위한 퍼블리셔 추가
                self.pub_image_blue_filtered = self.create_publisher(
                    Image, '/detect/image_output_sub4', 1)

        # 핵심 추가: 신호등 상태를 발행하는 퍼블리셔들
        self.pub_traffic_light_state = self.create_publisher(UInt8, '/detect/traffic_light', 1)
        self.pub_red_reliability = self.create_publisher(UInt8, '/detect/red_light_reliability', 1)
        self.pub_yellow_reliability = self.create_publisher(UInt8, '/detect/yellow_light_reliability', 1)
        self.pub_green_reliability = self.create_publisher(UInt8, '/detect/green_light_reliability', 1)

        self.cvBridge = CvBridge()
        self.cv_image = None

        self.is_image_available = False
        
        # 신호등 상태 정의 (detect_lane.py의 lane_state와 유사)
        # 0: 신호등 없음, 1: 빨간불, 2: 노란불, 3: 초록불
        self.TRAFFIC_LIGHT_NONE = 0
        self.TRAFFIC_LIGHT_RED = 1
        self.TRAFFIC_LIGHT_YELLOW = 2
        self.TRAFFIC_LIGHT_GREEN = 3

        # 신뢰도 관리 (detect_lane.py와 유사)
        self.reliability_red_light = 100
        self.reliability_yellow_light = 100
        self.reliability_green_light = 100

        # 감지 횟수를 통한 안정성 확보
        self.red_count = 0
        self.yellow_count = 0
        self.green_count = 0
        self.no_light_count = 0

        time.sleep(1)
        self.timer = self.create_timer(0.05, self.timer_callback)

    def get_detect_traffic_light_param(self, params):
        for param in params:
            if param.name == 'red.hue_l':
                self.hue_red_l = param.value
                self.get_logger().info(f'red.hue_l set to: {param.value}')
            elif param.name == 'red.hue_h':
                self.hue_red_h = param.value
                self.get_logger().info(f'red.hue_h set to: {param.value}')
            elif param.name == 'red.saturation_l':
                self.saturation_red_l = param.value
                self.get_logger().info(f'red.saturation_l set to: {param.value}')
            elif param.name == 'red.saturation_h':
                self.saturation_red_h = param.value
                self.get_logger().info(f'red.saturation_h set to: {param.value}')
            elif param.name == 'red.lightness_l':
                self.lightness_red_l = param.value
                self.get_logger().info(f'red.lightness_l set to: {param.value}')
            elif param.name == 'red.lightness_h':
                self.lightness_red_h = param.value
                self.get_logger().info(f'red.lightness_h set to: {param.value}')
            elif param.name == 'yellow.hue_l':
                self.hue_yellow_l = param.value
                self.get_logger().info(f'yellow.hue_l set to: {param.value}')
            elif param.name == 'yellow.hue_h':
                self.hue_yellow_h = param.value
                self.get_logger().info(f'yellow.hue_h set to: {param.value}')
            elif param.name == 'yellow.saturation_l':
                self.saturation_yellow_l = param.value
                self.get_logger().info(f'yellow.saturation_l set to: {param.value}')
            elif param.name == 'yellow.saturation_h':
                self.saturation_yellow_h = param.value
                self.get_logger().info(f'yellow.saturation_h set to: {param.value}')
            elif param.name == 'yellow.lightness_l':
                self.lightness_yellow_l = param.value
                self.get_logger().info(f'yellow.lightness_l set to: {param.value}')
            elif param.name == 'yellow.lightness_h':
                self.lightness_yellow_h = param.value
                self.get_logger().info(f'yellow.lightness_h set to: {param.value}')
            elif param.name == 'green.hue_l':
                self.hue_green_l = param.value
                self.get_logger().info(f'green.hue_l set to: {param.value}')
            elif param.name == 'green.hue_h':
                self.hue_green_h = param.value
                self.get_logger().info(f'green.hue_h set to: {param.value}')
            elif param.name == 'green.saturation_l':
                self.saturation_green_l = param.value
                self.get_logger().info(f'green.saturation_l set to: {param.value}')
            elif param.name == 'green.saturation_h':
                self.saturation_green_h = param.value
                self.get_logger().info(f'green.saturation_h set to: {param.value}')
            elif param.name == 'green.lightness_l':
                self.lightness_green_l = param.value
                self.get_logger().info(f'green.lightness_l set to: {param.value}')
            elif param.name == 'green.lightness_h':
                self.lightness_green_h = param.value
                self.get_logger().info(f'green.lightness_h set to: {param.value}')
        return SetParametersResult(successful=True)

    def get_image(self, image_msg):
        # detect_lane.py와 동일한 프레임 속도 제어
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            try:
                self.cv_image = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')
            except CvBridgeError as e:
                self.get_logger().error(f'CvBridge Error: {e}')
                return

        self.is_image_available = True

    def timer_callback(self):
        if self.is_image_available:
            self.find_traffic_light()

    def find_traffic_light(self):
        # 각 색상별 마스크 생성 및 감지
        cv_image_mask_red = self.mask_red_traffic_light()
        cv_image_mask_red = cv2.GaussianBlur(cv_image_mask_red, (5, 5), 0)
        detect_red = self.find_circle_of_traffic_light(cv_image_mask_red, 'red')

        cv_image_mask_yellow = self.mask_yellow_traffic_light()
        cv_image_mask_yellow = cv2.GaussianBlur(cv_image_mask_yellow, (5, 5), 0)
        detect_yellow = self.find_circle_of_traffic_light(cv_image_mask_yellow, 'yellow')

        cv_image_mask_green = self.mask_green_traffic_light()
        cv_image_mask_green = cv2.GaussianBlur(cv_image_mask_green, (5, 5), 0)
        detect_green = self.find_circle_of_traffic_light(cv_image_mask_green, 'green')

        # 감지 결과에 따른 카운터 업데이트 및 신뢰도 계산
        self.update_detection_counts(detect_red, detect_yellow, detect_green)
        self.calculate_reliability()
        
        # 최종 신호등 상태 결정 및 발행
        traffic_light_state = self.determine_traffic_light_state(detect_red, detect_yellow, detect_green)
        self.publish_traffic_light_state(traffic_light_state)

        # 시각화 (원본 코드 유지)
        if detect_red:
            cv2.putText(self.cv_image, 'RED', (self.point_x, self.point_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        if detect_yellow:
            cv2.putText(self.cv_image, 'YELLOW', (self.point_x, self.point_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255))
        if detect_green:
            cv2.putText(self.cv_image, 'GREEN', (self.point_x, self.point_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

        if self.pub_image_type == 'compressed':
            self.pub_image_traffic_light.publish(
                self.cvBridge.cv2_to_compressed_imgmsg(self.cv_image, 'jpg'))
        else:
            self.pub_image_traffic_light.publish(
                self.cvBridge.cv2_to_imgmsg(self.cv_image, 'bgr8'))

    def update_detection_counts(self, detect_red, detect_yellow, detect_green):
        """개선된 감지 카운터 업데이트 - 더 빠른 반응"""
        increment = 3  # 3 유지
        decrement = 1  # 유지
        
        # 상호 배타적 카운팅 (신호등은 한 번에 하나만 켜짐)
        detected_colors = sum([detect_red, detect_yellow, detect_green])
        
        if detected_colors <= 1:  # 하나 이하만 감지된 경우만 정상 처리
            if detect_red:
                self.red_count = min(10, self.red_count + increment)
                # 다른 색상 감지 시 빠른 감소
                self.yellow_count = max(0, self.yellow_count - increment)
                self.green_count = max(0, self.green_count - increment)
            elif detect_yellow:
                self.yellow_count = min(10, self.yellow_count + increment)
                self.red_count = max(0, self.red_count - increment)
                self.green_count = max(0, self.green_count - increment)
            elif detect_green:
                self.green_count = min(10, self.green_count + increment)
                self.red_count = max(0, self.red_count - increment)
                self.yellow_count = max(0, self.yellow_count - increment)
            else:
                # 아무것도 감지되지 않은 경우
                self.red_count = max(0, self.red_count - decrement)
                self.yellow_count = max(0, self.yellow_count - decrement)
                self.green_count = max(0, self.green_count - decrement)
                self.no_light_count += 1
        else:
            # 여러 색상이 동시에 감지된 경우 - 노이즈로 간주하고 감소
            self.red_count = max(0, self.red_count - decrement)
            self.yellow_count = max(0, self.yellow_count - decrement)
            self.green_count = max(0, self.green_count - decrement)

    def calculate_reliability(self):
        """각 색상별 신뢰도 계산 (detect_lane.py의 신뢰도 계산과 유사)"""
        reliability_change = 8  # 기존 5 → 8으로 증가
        # 빨간불 신뢰도
        if self.red_count >= 2:  # 2번 이상 연속 감지
            if self.reliability_red_light <= 92:
                self.reliability_red_light += reliability_change
        else:
            if self.reliability_red_light >=  reliability_change:
                self.reliability_red_light -=  reliability_change
                
        # 노란불 신뢰도
        if self.yellow_count >= 2:
            if self.reliability_yellow_light <= 92:
                self.reliability_yellow_light +=  reliability_change
        else:
            if self.reliability_yellow_light >=  reliability_change:
                self.reliability_yellow_light -=  reliability_change
                
        # 초록불 신뢰도
        if self.green_count >= 2:
            if self.reliability_green_light <= 92:
                self.reliability_green_light +=  reliability_change
        else:
            if self.reliability_green_light >=  reliability_change:
                self.reliability_green_light -=  reliability_change

        # 신뢰도 발행
        self.publish_reliability()

    def publish_reliability(self):
        """신뢰도 정보 발행"""
        msg_red_reliability = UInt8()
        msg_red_reliability.data = self.reliability_red_light
        self.pub_red_reliability.publish(msg_red_reliability)

        msg_yellow_reliability = UInt8()
        msg_yellow_reliability.data = self.reliability_yellow_light
        self.pub_yellow_reliability.publish(msg_yellow_reliability)

        msg_green_reliability = UInt8()
        msg_green_reliability.data = self.reliability_green_light
        self.pub_green_reliability.publish(msg_green_reliability)

    def determine_traffic_light_state(self, detect_red, detect_yellow, detect_green):
        """개선된 신호등 상태 결정 - 상호 배타적 로직"""
        
        # 1단계: 현재 프레임에서 실제 감지된 색상 확인
        current_detections = []
        if detect_red:
            current_detections.append('red')
        if detect_yellow:
            current_detections.append('yellow')
        if detect_green:
            current_detections.append('green')
        
        # 2단계: 신뢰도 기반 판단 (임계값 낮춤)
        reliable_red = self.reliability_red_light > 30 and self.red_count >= 2    # 50, 3 → 30, 2
        reliable_yellow = self.reliability_yellow_light > 30 and self.yellow_count >= 2
        reliable_green = self.reliability_green_light > 30 and self.green_count >= 2
        
        # 3단계: 상호 배타적 신호등 로직 (한 번에 하나만)
        if len(current_detections) > 1:
            # 여러 색상이 동시에 감지되면 우선순위로 결정
            if detect_red:
                return self.TRAFFIC_LIGHT_RED
            elif detect_yellow:
                return self.TRAFFIC_LIGHT_YELLOW
            elif detect_green:
                return self.TRAFFIC_LIGHT_GREEN
        
        # 4단계: 상태 변경 가속화 로직
        current_state = getattr(self, 'last_published_state', self.TRAFFIC_LIGHT_NONE)
        
        # 빨간불에서 다른 색상으로 변경 시 빠른 전환
        if current_state == self.TRAFFIC_LIGHT_RED:
            if detect_green and self.green_count >= 1:  # 더 빠른 전환
                return self.TRAFFIC_LIGHT_GREEN
            elif detect_yellow and self.yellow_count >= 1:
                return self.TRAFFIC_LIGHT_YELLOW
        
        # 초록불에서 다른 색상으로 변경 시 (안전 고려)
        if current_state == self.TRAFFIC_LIGHT_GREEN:
            if detect_red and self.red_count >= 1:  # 안전을 위해 빠른 전환
                return self.TRAFFIC_LIGHT_RED
            elif detect_yellow and self.yellow_count >= 1:
                return self.TRAFFIC_LIGHT_YELLOW
        
        # 5단계: 일반적인 신뢰도 기반 판단
        if reliable_red:
            return self.TRAFFIC_LIGHT_RED
        elif reliable_yellow:
            return self.TRAFFIC_LIGHT_YELLOW
        elif reliable_green:
            return self.TRAFFIC_LIGHT_GREEN
        else:
            return self.TRAFFIC_LIGHT_NONE

    def publish_traffic_light_state(self, state):
        msg_traffic_light_state = UInt8()
        msg_traffic_light_state.data = state
        self.pub_traffic_light_state.publish(msg_traffic_light_state)

        # 상태 변경 시에만 로그
        if not hasattr(self, 'last_published_state'):
            self.last_published_state = None
        
        if self.last_published_state != state:
            state_names = {0: 'NONE', 1: 'RED', 2: 'YELLOW', 3: 'GREEN'}
            self.get_logger().info(f'🚦 Traffic Light: {state_names.get(state, "UNKNOWN")}')
            self.last_published_state = state

    def mask_red_traffic_light(self):
        image = np.copy(self.cv_image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red = np.array([self.hue_red_l, self.saturation_red_l, self.lightness_red_l])
        upper_red = np.array([self.hue_red_h, self.saturation_red_h, self.lightness_red_h])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_red_light.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg'))
            else:
                self.pub_image_red_light.publish(
                    self.cvBridge.cv2_to_imgmsg(mask, 'mono8'))

        mask = cv2.bitwise_not(mask)
        return mask

    def mask_yellow_traffic_light(self):
        image = np.copy(self.cv_image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array(
            [self.hue_yellow_l, self.saturation_yellow_l, self.lightness_yellow_l]
            )
        upper_yellow = np.array(
            [self.hue_yellow_h, self.saturation_yellow_h, self.lightness_yellow_h]
            )

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_yellow_light.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg'))
            else:
                self.pub_image_yellow_light.publish(
                    self.cvBridge.cv2_to_imgmsg(mask, 'mono8'))

        mask = cv2.bitwise_not(mask)
        return mask

    def mask_green_traffic_light(self):
        """개선된 초록색 마스킹 - 파란색 표지판 완전 제거"""
        image = np.copy(self.cv_image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 1. 초록색 기본 범위
        lower_green = np.array([self.hue_green_l, self.saturation_green_l, self.lightness_green_l])
        upper_green = np.array([self.hue_green_h, self.saturation_green_h, self.lightness_green_h])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # 2. 파란색 표지판 영역 감지 및 제거 (사용자 분석 결과 적용)
        blue_lower = np.array([self.BLUE_SIGN_H_MIN, self.BLUE_SIGN_S_MIN, self.BLUE_SIGN_V_MIN])
        blue_upper = np.array([self.BLUE_SIGN_H_MAX, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # 파란색 영역을 초록색 마스크에서 완전 제거
        green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(blue_mask))
        
        # 3. 추가 필터링: 형태학적 연산
        kernel = np.ones((3, 3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        
        # 4. ROI 영역 제한 (신호등 위치)
        height, width = green_mask.shape[:2]
        roi_mask = np.zeros_like(green_mask)
        roi_x_start = width // 2
        roi_x_end = width
        roi_y_start = height // 4
        roi_y_end = 3 * height // 4
        roi_mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = 255
        green_mask = cv2.bitwise_and(green_mask, roi_mask)
        
        # 캘리브레이션 모드에서 결과 확인
        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_green_light.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(green_mask, 'jpg'))
                # 파란색 필터링 결과도 발행
                blue_filtered_result = cv2.bitwise_not(blue_mask)
                self.pub_image_blue_filtered.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(blue_filtered_result, 'jpg'))
            else:
                self.pub_image_green_light.publish(
                    self.cvBridge.cv2_to_imgmsg(green_mask, 'mono8'))
                # 파란색 필터링 결과도 발행
                blue_filtered_result = cv2.bitwise_not(blue_mask)
                self.pub_image_blue_filtered.publish(
                    self.cvBridge.cv2_to_imgmsg(blue_filtered_result, 'mono8'))
        
        # SimpleBlobDetector를 위해 반전
        green_mask = cv2.bitwise_not(green_mask)
        return green_mask

    def find_circle_of_traffic_light(self, mask, color):
        detect_result = False
        
        height, width = mask.shape[:2]
        roi_x_start = width // 2
        roi_x_end = width
        roi_y_start = height // 3
        roi_y_end = 2 * height // 3

        # ROI 영역만으로 마스크 제한
        roi_mask = np.zeros_like(mask)
        roi_mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        # 노이즈 제거 및 정제
        kernel = np.ones((5, 5), np.uint8)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
        # HoughCircles를 사용한 원형 감지
        # 마스크를 반전 (HoughCircles는 밝은 원을 찾음)
        inverted_mask = cv2.bitwise_not(roi_mask)
        
        # 가우시안 블러 적용
        blurred = cv2.GaussianBlur(inverted_mask, (11, 11), 2)
        
        # HoughCircles로 원 감지
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,      # 원 중심 간 최소 거리
            param1=50,       # Canny 상위 임계값
            param2=20,       # 누적기 임계값 (낮을수록 더 많이 검출)
            minRadius=6,     # 신호등 크기에 맞게 조정
            maxRadius=40     # 신호등 크기에 맞게 조정
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # 첫 번째 원 사용 (가장 확실한 것)
            if len(circles) > 0:
                self.point_x = circles[0][0]
                self.point_y = circles[0][1]
                detect_result = True
                
                # 디버깅을 위한 로그
                # self.get_logger().info(f'{color} circle detected at ({self.point_x}, {self.point_y}) with radius {circles[0][2]}')
        
        return detect_result


def main(args=None):
    rclpy.init(args=args)
    node = DetectTrafficLight()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()