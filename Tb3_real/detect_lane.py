#!/usr/bin/env python3
#
# Optimized Lane Detection with History-based Virtual Center + Lane Center Clamping
# Based on virtual centerline generation + clamping techniques
#

import cv2
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_msgs.msg import UInt8
import time


class DetectLineGeometric(Node):

    def __init__(self):
        super().__init__('detect_line_geometric')

        # ROS2 파라미터 설정
        parameter_descriptor_int = ParameterDescriptor(
            description='Integer parameter range',
            integer_range=[IntegerRange(from_value=0, to_value=500, step=1)]
        )
        parameter_descriptor_float = ParameterDescriptor(
            description='Float parameter range'
        )
        
        self.declare_parameters(
            namespace='',
            parameters=[
                # 기본 이미지 처리 파라미터
                ('image.roi_ratio', 0.9, parameter_descriptor_float),
                ('image.hough_threshold', 60, parameter_descriptor_int),
                
                # 차선 필터링 파라미터
                ('filtering.min_line_length', 50, parameter_descriptor_int),
                ('filtering.max_line_gap', 30, parameter_descriptor_int),
                ('filtering.min_slope', 0.05, parameter_descriptor_float),
                ('filtering.max_slope', 112.0, parameter_descriptor_float),
                ('filtering.enable_thickness_check', True),
                ('filtering.min_thick_pixels', 10, parameter_descriptor_int),
                ('filtering.max_thick_pixels', 19, parameter_descriptor_int),
                
                # 중심선 계산 파라미터
                ('center.single_lane_offset_ratio', 0.25, parameter_descriptor_float),
                ('center.slope_position_tolerance', 50, parameter_descriptor_int),
                
                # 히스토리 기반 가상 중심선 파라미터
                ('history.enable_virtual_center', True),
                ('history.lane_width_history_size', 10, parameter_descriptor_int),
                ('history.default_lane_width', 150, parameter_descriptor_int),
                
                # Lane Center Clamping 파라미터
                ('clamping.enable_clamping', True),
                ('clamping.max_move_distance', 15.0, parameter_descriptor_float),
                
                # 시스템 파라미터
                ('system.calibration_mode', False)
            ]
        )

        # 파라미터 값 가져오기
        self.roi_ratio = self.get_parameter('image.roi_ratio').get_parameter_value().double_value
        self.hough_threshold = self.get_parameter('image.hough_threshold').get_parameter_value().integer_value
        
        self.min_line_length = self.get_parameter('filtering.min_line_length').get_parameter_value().integer_value
        self.max_line_gap = self.get_parameter('filtering.max_line_gap').get_parameter_value().integer_value
        self.min_slope = self.get_parameter('filtering.min_slope').get_parameter_value().double_value
        self.max_slope = self.get_parameter('filtering.max_slope').get_parameter_value().double_value
        self.enable_thickness_check = self.get_parameter('filtering.enable_thickness_check').get_parameter_value().bool_value
        self.min_thick_pixels = self.get_parameter('filtering.min_thick_pixels').get_parameter_value().integer_value
        self.max_thick_pixels = self.get_parameter('filtering.max_thick_pixels').get_parameter_value().integer_value
        
        self.single_lane_offset_ratio = self.get_parameter('center.single_lane_offset_ratio').get_parameter_value().double_value
        self.slope_position_tolerance = self.get_parameter('center.slope_position_tolerance').get_parameter_value().integer_value
        
        self.enable_virtual_center = self.get_parameter('history.enable_virtual_center').get_parameter_value().bool_value
        self.history_size = self.get_parameter('history.lane_width_history_size').get_parameter_value().integer_value
        self.default_lane_width = self.get_parameter('history.default_lane_width').get_parameter_value().integer_value
        
        self.enable_clamping = self.get_parameter('clamping.enable_clamping').get_parameter_value().bool_value
        self.max_move_distance = self.get_parameter('clamping.max_move_distance').get_parameter_value().double_value
        
        self.is_calibration_mode = self.get_parameter('system.calibration_mode').get_parameter_value().bool_value

        if self.is_calibration_mode:
            self.add_on_set_parameters_callback(self.cbGetDetectLineParam)

        # 이미지 구독/발행 설정
        self.sub_image_type = 'raw'
        self.pub_image_type = 'compressed'

        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage, '/detect/image_input/compressed', self.cbProcessImage, 1)
        elif self.sub_image_type == 'raw':
            self.sub_image_original = self.create_subscription(
                Image, '/detect/image_input', self.cbProcessImage, 1)

        if self.pub_image_type == 'compressed':
            self.pub_image_lane = self.create_publisher(
                CompressedImage, '/detect/image_output/compressed', 1)
        elif self.pub_image_type == 'raw':
            self.pub_image_lane = self.create_publisher(
                Image, '/detect/image_output', 1)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_left = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub1/compressed', 1)
                self.pub_image_right = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub2/compressed', 1)
            elif self.pub_image_type == 'raw':
                self.pub_image_left = self.create_publisher(
                    Image, '/detect/image_output_sub1', 1)
                self.pub_image_right = self.create_publisher(
                    Image, '/detect/image_output_sub2', 1)

        # 주요 출력 토픽 설정
        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 1)
        self.pub_lane_state = self.create_publisher(UInt8, '/detect/lane_state', 1)

        self.cvBridge = CvBridge()
        self.counter = 1

        # 히스토리 기반 가상 중심선 변수
        self.prev_lane_width = self.default_lane_width
        self.lane_width_history = []
        
        # Lane Center Clamping 변수
        self.previous_lane_center = None
        
        # 성능 추적 (시각화용)
        self.fps_count = 0
        self.fps_time = time.time()
        self.current_fps = 0

        # 전처리된 엣지 이미지 저장 (두께 검사용)
        self.current_edges = None

    def cbGetDetectLineParam(self, parameters):
        """캘리브레이션 모드에서 파라미터 변경 시 호출되는 콜백"""
        for param in parameters:
            self.get_logger().info(f'Parameter: {param.name} = {param.value}')
            
            if param.name == 'image.roi_ratio':
                self.roi_ratio = param.value
            elif param.name == 'image.hough_threshold':
                self.hough_threshold = param.value
            elif param.name == 'filtering.min_line_length':
                self.min_line_length = param.value
            elif param.name == 'filtering.max_line_gap':
                self.max_line_gap = param.value
            elif param.name == 'filtering.min_slope':
                self.min_slope = param.value
            elif param.name == 'filtering.max_slope':
                self.max_slope = param.value
            elif param.name == 'filtering.enable_thickness_check':
                self.enable_thickness_check = param.value
            elif param.name == 'filtering.min_thick_pixels':
                self.min_thick_pixels = param.value
            elif param.name == 'filtering.max_thick_pixels':
                self.max_thick_pixels = param.value
            elif param.name == 'center.single_lane_offset_ratio':
                self.single_lane_offset_ratio = param.value
            elif param.name == 'center.slope_position_tolerance':
                self.slope_position_tolerance = param.value
            elif param.name == 'history.enable_virtual_center':
                self.enable_virtual_center = param.value
            elif param.name == 'history.lane_width_history_size':
                self.history_size = param.value
            elif param.name == 'history.default_lane_width':
                self.default_lane_width = param.value
            elif param.name == 'clamping.enable_clamping':
                self.enable_clamping = param.value
            elif param.name == 'clamping.max_move_distance':
                self.max_move_distance = param.value
                
        return SetParametersResult(successful=True)

    def cbProcessImage(self, image_msg):
        """메인 이미지 처리 콜백"""
        # 프레임 스킵 (3프레임마다 처리)
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        # ROS 이미지 메시지를 OpenCV 이미지로 변환
        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif self.sub_image_type == 'raw':
            cv_image = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')

        # 프레임 처리
        result, left_lines, right_lines = self.process_frame(cv_image)
        
        # 차선 중심 계산 및 퍼블리시
        self.calculate_and_publish_lane_center(left_lines, right_lines, cv_image.shape[1], cv_image.shape[0])
        
        # 처리된 이미지 퍼블리시
        if self.pub_image_type == 'compressed':
            self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(result, 'jpg'))
        elif self.pub_image_type == 'raw':
            self.pub_image_lane.publish(self.cvBridge.cv2_to_imgmsg(result, 'bgr8'))

        # 캘리브레이션 모드에서 추가 이미지 퍼블리시
        if self.is_calibration_mode:
            self.publish_calibration_images(cv_image, left_lines, right_lines)

    def apply_clamping(self, raw_center):
        """차선 중심 클램핑 - 최대 이동 거리 제한 적용"""
        if not self.enable_clamping or self.previous_lane_center is None:
            self.previous_lane_center = raw_center
            return raw_center
        
        # 이동 거리 계산 및 제한
        distance = raw_center - self.previous_lane_center
        if abs(distance) > self.max_move_distance:
            if distance > 0:
                clamped_center = self.previous_lane_center + self.max_move_distance
            else:
                clamped_center = self.previous_lane_center - self.max_move_distance
            self.get_logger().info(f'Clamping: {raw_center:.1f} -> {clamped_center:.1f} (move: {distance:.1f} -> {self.max_move_distance:.1f})')
        else:
            clamped_center = raw_center
        
        self.previous_lane_center = clamped_center
        return clamped_center

    def calculate_and_publish_lane_center(self, left_lines, right_lines, img_width, img_height):
        """차선 중심 계산 - 히스토리 기반 가상 중심선 및 클램핑 적용"""
        center_x = None
        lane_state = UInt8()
        virtual_center_line = None
        
        # 참조 y 위치 (하단 70% 기준)
        ref_y = int(img_height * 0.7)
        
        # 평균 x 위치 계산
        cx_left = None
        cx_right = None
        if left_lines:
            cx_left = np.mean([self.get_line_x_position(line, ref_y) for line in left_lines])
        if right_lines:
            cx_right = np.mean([self.get_line_x_position(line, ref_y) for line in right_lines])
        
        # 1. 양쪽 차선이 모두 있는 경우
        if cx_left is not None and cx_right is not None:
            center_x = (cx_left + cx_right) / 2
            lane_state.data = 2
            
            # 히스토리 업데이트
            if self.enable_virtual_center:
                lane_width = abs(cx_right - cx_left)
                self.update_lane_width_history(lane_width)
                
        # 2. 한쪽 차선만 있는 경우 - 가상 중심선 또는 고정 비율 보정
        elif cx_left is not None or cx_right is not None:
            if self.enable_virtual_center:
                # 가상 중심선 생성
                if cx_left is not None and not cx_right:
                    virtual_center_line = self.create_virtual_center_from_left(left_lines, img_height)
                    lane_state.data = 1
                elif cx_right is not None and not cx_left:
                    virtual_center_line = self.create_virtual_center_from_right(right_lines, img_height)
                    lane_state.data = 3
                    
                if virtual_center_line is not None:
                    center_x = self.get_line_x_position(virtual_center_line, ref_y)
            else:
                # 고정 비율 보정
                if cx_left is not None:
                    center_x = cx_left + (img_width * self.single_lane_offset_ratio)
                    lane_state.data = 1
                elif cx_right is not None:
                    center_x = cx_right - (img_width * self.single_lane_offset_ratio)
                    lane_state.data = 3
        else:
            # 3. 차선이 없는 경우
            lane_state.data = 0
        
        # 차선 상태 퍼블리시
        self.pub_lane_state.publish(lane_state)
        
        if center_x is not None:
            # 클램핑 적용
            clamped_center_x = self.apply_clamping(center_x)
            
            # 결과 퍼블리시
            msg_lane_center = Float64()
            msg_lane_center.data = float(clamped_center_x)
            self.pub_lane.publish(msg_lane_center)
            
            mode = "VIRTUAL" if virtual_center_line is not None else "NORMAL"
            clamping_info = "CLAMPED" if abs(clamped_center_x - center_x) > 0.1 else "RAW"
            
            self.get_logger().info(
                f'Lane center: {clamped_center_x:.1f} (raw: {center_x:.1f}), '
                f'State: {lane_state.data}, Mode: {mode}, {clamping_info}'
            )

    def update_lane_width_history(self, width):
        """차선 폭 히스토리 업데이트"""
        self.lane_width_history.append(width)
        if len(self.lane_width_history) > self.history_size:
            self.lane_width_history.pop(0)
        
        if self.lane_width_history:
            self.prev_lane_width = sum(self.lane_width_history) / len(self.lane_width_history)

    def create_virtual_center_from_right(self, right_lines, img_height):
        """오른쪽 차선을 기반으로 가상 중심선 생성"""
        if not right_lines:
            return None
        
        rightmost_line = min(right_lines, 
                           key=lambda line: self.get_line_x_position(line, img_height - img_height//4))
        
        x1, y1, x2, y2 = rightmost_line
        dx = x2 - x1
        dy = y2 - y1
        
        half_width = self.prev_lane_width / 2
        
        # 라인에 수직인 벡터 계산하여 왼쪽으로 이동
        if dy != 0:
            normal_x = -dy / np.sqrt(dx*dx + dy*dy) * half_width
            normal_y = dx / np.sqrt(dx*dx + dy*dy) * half_width
        else:
            normal_x = -half_width
            normal_y = 0
        
        virtual_center = [
            int(x1 + normal_x), int(y1 + normal_y),
            int(x2 + normal_x), int(y2 + normal_y)
        ]
        
        return virtual_center

    def create_virtual_center_from_left(self, left_lines, img_height):
        """왼쪽 차선을 기반으로 가상 중심선 생성"""
        if not left_lines:
            return None
        
        leftmost_line = max(left_lines, 
                          key=lambda line: self.get_line_x_position(line, img_height - img_height//4))
        
        x1, y1, x2, y2 = leftmost_line
        dx = x2 - x1
        dy = y2 - y1
        
        half_width = self.prev_lane_width / 2
        
        # 라인에 수직인 벡터 계산하여 오른쪽으로 이동
        if dy != 0:
            normal_x = -dy / np.sqrt(dx*dx + dy*dy) * half_width
            normal_y = -dx / np.sqrt(dx*dx + dy*dy) * half_width
        else:
            normal_x = half_width
            normal_y = 0
        
        virtual_center = [
            int(x1 + normal_x), int(y1 + normal_y),
            int(x2 + normal_x), int(y2 + normal_y)
        ]
        
        return virtual_center

    def publish_calibration_images(self, original_img, left_lines, right_lines):
        """캘리브레이션용 이미지 퍼블리시"""
        # 왼쪽 차선 시각화
        left_img = original_img.copy()
        for line in left_lines:
            x1, y1, x2, y2 = line
            cv2.line(left_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 오른쪽 차선 시각화
        right_img = original_img.copy()
        for line in right_lines:
            x1, y1, x2, y2 = line
            cv2.line(right_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
        # 이미지 퍼블리시
        if self.pub_image_type == 'compressed':
            self.pub_image_left.publish(self.cvBridge.cv2_to_compressed_imgmsg(left_img, 'jpg'))
            self.pub_image_right.publish(self.cvBridge.cv2_to_compressed_imgmsg(right_img, 'jpg'))
        elif self.pub_image_type == 'raw':
            self.pub_image_left.publish(self.cvBridge.cv2_to_imgmsg(left_img, 'bgr8'))
            self.pub_image_right.publish(self.cvBridge.cv2_to_imgmsg(right_img, 'bgr8'))

    def preprocess_image(self, img):
        """이미지 전처리"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 80, 200)
        
        # 모폴로지 연산으로 엣지 연결 및 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        final_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        self.current_edges = final_edges
        return final_edges

    def get_roi(self, img):
        """ROI 마스크 적용"""
        height, width = img.shape
        roi_top = int(height * (1 - self.roi_ratio))
        
        mask = np.zeros_like(img)
        mask[roi_top:, :] = 255
        
        return cv2.bitwise_and(img, mask)

    def filter_lines(self, lines):
        """검출된 직선 필터링"""
        if lines is None:
            return []
        
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 1. 길이 필터링
            line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if line_length < self.min_line_length:
                continue
            
            # 2. 기울기 필터링
            dx = x2 - x1
            dy = y2 - y1
            
            if abs(dx) < 1e-6:
                slope = float('inf')
            else:
                slope = abs(dy / dx)
            
            if slope < self.min_slope or slope > self.max_slope:
                continue
            
            # 3. 두께 검사 (활성화된 경우)
            if self.enable_thickness_check:
                if not self.check_line_thickness(x1, y1, x2, y2):
                    continue
            
            filtered_lines.append(line[0])
        
        return filtered_lines

    def check_line_thickness(self, x1, y1, x2, y2):
        """선 두께 검사"""
        if self.current_edges is None:
            return True
            
        dx = x2 - x1
        dy = y2 - y1
        
        if abs(dx) < 1e-6:
            slope = float('inf') if dy > 0 else float('-inf')
        else:
            slope = dy / dx
        
        num_thick_pixels = 0
        points = np.linspace((x1, y1), (x2, y2), 10).astype(int)
        
        for px, py in points:
            for offset in range(-2, 3):
                if abs(slope) == float('inf'):
                    ox = px + offset
                    oy = py
                else:
                    norm_factor = np.sqrt(1 + slope**2) + 1e-6
                    ox = int(px - offset * slope / norm_factor)
                    oy = int(py + offset / norm_factor)
                
                if (0 <= oy < self.current_edges.shape[0] and 
                    0 <= ox < self.current_edges.shape[1]):
                    if self.current_edges[oy, ox] > 0:
                        num_thick_pixels += 1
        
        return self.min_thick_pixels <= num_thick_pixels <= self.max_thick_pixels

    def classify_lines(self, filtered_lines, img_width):
        """필터링된 직선을 왼쪽/오른쪽 차선으로 분류"""
        left_lines = []
        right_lines = []
        
        if not filtered_lines:
            return left_lines, right_lines
        
        mid_x_original = img_width // 2
        ref_y = img_width - img_width // 4
        
        for line in filtered_lines:
            x1, y1, x2, y2 = line
            
            dx = x2 - x1
            dy = y2 - y1
            
            if abs(dx) < 1e-6:
                continue
                
            slope = dy / dx
            line_center_x = self.get_line_x_position(line, ref_y)
            
            if (slope < 0 and line_center_x < mid_x_original + self.slope_position_tolerance):
                left_lines.append(line)
            elif (slope > 0 and line_center_x > mid_x_original - self.slope_position_tolerance):
                right_lines.append(line)
        
        return left_lines, right_lines

    def get_line_x_position(self, line, y_ref):
        """주어진 y 좌표에서 라인의 x 위치 계산"""
        x1, y1, x2, y2 = line
        
        if y2 == y1:
            return (x1 + x2) / 2
        
        slope = (x2 - x1) / (y2 - y1)
        x_at_y = x1 + slope * (y_ref - y1)
        
        return x_at_y

    def process_frame(self, frame):
        """메인 프레임 처리"""
        # 1. 전처리
        edges = self.preprocess_image(frame)
        roi = self.get_roi(edges)
        
        # 2. 허프 변환으로 직선 검출
        lines = cv2.HoughLinesP(roi, 
                               rho=1, 
                               theta=np.pi/180, 
                               threshold=self.hough_threshold,
                               minLineLength=self.min_line_length,
                               maxLineGap=self.max_line_gap)
        
        # 3. 필터링 및 분류
        filtered_lines = self.filter_lines(lines)
        left_lines, right_lines = self.classify_lines(filtered_lines, frame.shape[1])
        
        # 4. 결과 시각화
        result = self.draw_result(frame, left_lines, right_lines)
        
        return result, left_lines, right_lines

    def draw_result(self, img, left_lines, right_lines):
        """처리 결과 시각화"""
        result = img.copy()
        line_image = np.zeros_like(img)
        
        # 왼쪽 라인들 (초록색)
        for line in left_lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 오른쪽 라인들 (파란색)
        for line in right_lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 가상 중심선 표시 (있는 경우)
        ref_y = int(img.shape[0] * 0.7)
        cx_left = None
        cx_right = None
        
        if left_lines:
            cx_left = np.mean([self.get_line_x_position(line, ref_y) for line in left_lines])
        if right_lines:
            cx_right = np.mean([self.get_line_x_position(line, ref_y) for line in right_lines])
        
        # 가상 중심선 그리기
        if self.enable_virtual_center and (cx_left is not None) != (cx_right is not None):
            if cx_left is not None and cx_right is None:
                virtual_line = self.create_virtual_center_from_left(left_lines, img.shape[0])
            elif cx_right is not None and cx_left is None:
                virtual_line = self.create_virtual_center_from_right(right_lines, img.shape[0])
            else:
                virtual_line = None
                
            if virtual_line is not None:
                cv2.line(line_image, (virtual_line[0], virtual_line[1]), 
                        (virtual_line[2], virtual_line[3]), (255, 0, 255), 3)
        
        # 현재 중심선 표시 (클램핑된 값)
        if self.previous_lane_center is not None:
            cv2.line(line_image, (int(self.previous_lane_center), img.shape[0]), 
                    (int(self.previous_lane_center), int(img.shape[0] * 0.5)), (0, 0, 255), 2)
        
        # 이미지 합성
        result = cv2.addWeighted(img, 0.8, line_image, 1, 0)
        
        # 정보 텍스트 추가
        self.draw_info_text(result, left_lines, right_lines)
        
        return result

    def draw_info_text(self, img, left_lines, right_lines):
        """정보 텍스트 그리기"""
        # FPS 계산
        current_time = time.time()
        self.fps_count += 1
        
        if current_time - self.fps_time >= 1.0:
            self.current_fps = self.fps_count / (current_time - self.fps_time)
            self.fps_count = 0
            self.fps_time = current_time
        
        # 텍스트 정보
        cv2.putText(img, f'FPS: {self.current_fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        mode_text = 'MODE: HISTORY + CLAMP'
        if self.enable_virtual_center:
            mode_text += ' (VIRTUAL ON)'
        else:
            mode_text += ' (VIRTUAL OFF)'
            
        cv2.putText(img, mode_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f'Left: {len(left_lines)} | Right: {len(right_lines)}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 검출 상태
        virtual_line_active = (self.enable_virtual_center and 
                             (len(left_lines) > 0) != (len(right_lines) > 0))
        
        if len(left_lines) > 0 and len(right_lines) > 0:
            status = "BOTH LANES"
            color = (0, 255, 255)
        elif len(left_lines) > 0:
            status = "LEFT ONLY"
            if virtual_line_active:
                status += " + VIRTUAL"
            color = (0, 255, 0)
        elif len(right_lines) > 0:
            status = "RIGHT ONLY"
            if virtual_line_active:
                status += " + VIRTUAL"
            color = (255, 0, 0)
        else:
            status = "NO LANES"
            color = (0, 0, 255)
            
        cv2.putText(img, f'Status: {status}', (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 히스토리 정보
        if self.enable_virtual_center:
            cv2.putText(img, f'Lane Width: {self.prev_lane_width:.1f} (History: {len(self.lane_width_history)})', 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # 클램핑 정보
        if self.enable_clamping:
            clamping_text = f'Clamping: Max Move {self.max_move_distance:.1f}px'
            if self.previous_lane_center is not None:
                clamping_text += f' (Current: {self.previous_lane_center:.1f})'
            cv2.putText(img, clamping_text, (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # ROI 라인
        roi_line = int(img.shape[0] * (1 - self.roi_ratio))
        cv2.line(img, (0, roi_line), (img.shape[1], roi_line), (255, 255, 0), 2)


def main(args=None):
    rclpy.init(args=args)
    node = DetectLineGeometric()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()