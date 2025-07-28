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
# Author: Leon Jung, Gilbert, Ashe Kim, Jun
#
# ============================================================================
# 🤖 자율주행 로봇용 교통 표지판 인식 시스템 (거리 필터링 개선 버전)
# 
# 개선 사항:
# 1. 📏 표지판 크기 기반 거리 추정 및 필터링
# 2. 🔍 더 엄격한 매칭 조건으로 오감지 방지
# 3. 📊 연속 프레임 검증으로 안정성 향상
# 4. 🎯 적응형 임계값으로 정확도 개선
# ============================================================================

from enum import Enum
import os

import cv2                          # OpenCV: 컴퓨터 비전 라이브러리
from cv_bridge import CvBridge      # ROS ↔ OpenCV 이미지 형식 변환기
import numpy as np                  # NumPy: 수치 계산 및 배열 처리
import rclpy                        # ROS2 Python 클라이언트 라이브러리
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage  # 압축 이미지 메시지 타입
from sensor_msgs.msg import Image           # 원본 이미지 메시지 타입
from std_msgs.msg import UInt8              # 부호없는 8비트 정수 메시지
from std_msgs.msg import Bool               # 참/거짓 불린 메시지

from ament_index_python.packages import get_package_share_directory

class DetectSign(Node):
    """
    🎯 교통 표지판 감지 및 분류 클래스 (거리 필터링 개선 버전)
    
    주요 개선 사항:
    - 멀리 있는 표지판 필터링으로 근거리 표지판만 인식
    - 더 엄격한 검증 조건으로 오감지 방지
    - 연속 프레임 검증으로 안정성 향상
    """

    def __init__(self):
        super().__init__('detect_sign')  # ROS2 노드 이름 설정

        # =============================================================
        # 📊 이미지 처리 형식 설정
        # =============================================================
        self.sub_image_type = 'raw'         # 입력: 'compressed'(압축) 또는 'raw'(원본)
        self.pub_image_type = 'compressed'  # 출력: 'compressed'(압축) 또는 'raw'(원본)

        # =============================================================
        # 🎯 거리 필터링 및 검증 강화 설정
        # =============================================================
        
        # 표지판 크기 기반 거리 필터링 (픽셀 기준)
        self.MIN_SIGN_AREA = 2000      # 최소 표지판 영역 (픽셀²) - 너무 작은 표지판 무시
        self.MAX_SIGN_AREA = 50000     # 최대 표지판 영역 (픽셀²) - 너무 큰 표지판 무시
        self.MIN_SIGN_WIDTH = 40       # 최소 표지판 너비 (픽셀)
        self.MIN_SIGN_HEIGHT = 40      # 최소 표지판 높이 (픽셀)
        
        # 연속 프레임 검증을 위한 카운터
        self.detection_history = {
            'stop': 0,
            'left': 0, 
            'right': 0
        }
        self.DETECTION_THRESHOLD = 3   # 3번 연속 감지되어야 유효한 것으로 판단
        self.MAX_HISTORY = 5          # 최대 히스토리 길이
        
        # 마지막 감지 시간 추적 (중복 발행 방지)
        self.last_detection_time = {
            'stop': 0,
            'left': 0,
            'right': 0
        }
        self.MIN_DETECTION_INTERVAL = 5.0  # 최소 2초 간격으로만 감지 신호 발행

        # =============================================================
        # 📥 구독자(Subscriber) 설정 - 카메라 데이터 수신
        # =============================================================
        
        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage,
                '/detect/image_input/compressed',
                self.cbFindTrafficSign,
                10
            )
        elif self.sub_image_type == 'raw':
            self.sub_image_original = self.create_subscription(
                Image,
                '/detect/image_input',
                self.cbFindTrafficSign,
                10
            )

        # =============================================================
        # 📤 발행자(Publisher) 설정 - 감지 결과 송신
        # =============================================================
        
        self.pub_traffic_sign = self.create_publisher(UInt8, '/detect/traffic_sign', 10)
        
        if self.pub_image_type == 'compressed':
            self.pub_image_traffic_sign = self.create_publisher(
                CompressedImage,
                '/detect/image_output/compressed', 10
            )
        elif self.pub_image_type == 'raw':
            self.pub_image_traffic_sign = self.create_publisher(
                Image, '/detect/image_output', 10
            )

        self.pub_stop_detected = self.create_publisher(Bool, '/detect/stop_sign', 10)
        self.pub_left_detected = self.create_publisher(Bool, '/detect/left_sign', 10)              
        self.pub_right_detected = self.create_publisher(Bool, '/detect/right_sign', 10)

        # =============================================================
        # 🔧 시스템 구성 요소 초기화
        # =============================================================
        self.cvBridge = CvBridge()
        self.TrafficSign = Enum('TrafficSign', 'stop left right')
        self.counter = 1

        # 컴퓨터 비전 알고리즘 초기화 및 템플릿 로드
        self.fnPreproc()

        self.get_logger().info('🚀 Precision DetectSign Node Initialized with Advanced Filtering!')

    def fnPreproc(self):
        """
        🔬 컴퓨터 비전 시스템 전처리 및 초기화
        """
        
        # SIFT 검출기 생성 (더 엄격한 설정)
        self.sift = cv2.SIFT_create(
            nfeatures=500,        # 최대 특징점 수 제한
            contrastThreshold=0.04,  # 대비 임계값 (기본값보다 높게 설정)
            edgeThreshold=15      # 가장자리 임계값 (기본값보다 높게 설정)
        )
        
        # 템플릿 이미지 로드
        package_share_dir = get_package_share_directory('turtlebot3_autorace_detect')

        self.img_stop = cv2.imread(os.path.join(package_share_dir, 'image', 'stop.png'), cv2.IMREAD_GRAYSCALE)
        self.img_left = cv2.imread(os.path.join(package_share_dir, 'image', 'left.png'), cv2.IMREAD_GRAYSCALE)
        self.img_right = cv2.imread(os.path.join(package_share_dir, 'image', 'right.png'), cv2.IMREAD_GRAYSCALE)

        # 이미지 로드 실패 시 프로그램 안전 종료
        if self.img_stop is None or self.img_left is None or self.img_right is None:
            self.get_logger().error("Failed to load template images")
            rclpy.shutdown()
            return

        # 템플릿 이미지 크기 저장 (거리 추정용)
        self.template_sizes = {
            'stop': self.img_stop.shape,
            'left': self.img_left.shape,
            'right': self.img_right.shape
        }

        # 템플릿 특징점 미리 계산
        self.kp_stop, self.des_stop = self.sift.detectAndCompute(self.img_stop, None)
        self.kp_left, self.des_left = self.sift.detectAndCompute(self.img_left, None)
        self.kp_right, self.des_right = self.sift.detectAndCompute(self.img_right, None)
        
        # FLANN 매처 설정
        FLANN_INDEX_KDTREE = 0
        index_params = {
            'algorithm': FLANN_INDEX_KDTREE,
            'trees': 5
        }
        search_params = {
            'checks': 50
        }
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def fnCalcMSE(self, arr1, arr2):
        """
        📊 MSE (평균 제곱 오차) 계산 함수
        """
        squared_diff = (arr1 - arr2) ** 2
        total_sum = np.sum(squared_diff)
        num_all = arr1.shape[0] * arr1.shape[1]
        err = total_sum / num_all
        return err

    def fnCalculateSignSize(self, src_pts, dst_pts, template_shape):
        """
        🔍 표지판 크기 계산 및 거리 추정 함수
        
        Args:
            src_pts: 입력 이미지의 매칭점들
            dst_pts: 템플릿 이미지의 매칭점들  
            template_shape: 템플릿 이미지 크기
            
        Returns:
            tuple: (면적, 너비, 높이, 스케일 비율)
        """
        
        # 바운딩 박스 계산
        src_points = src_pts.reshape(-1, 2)
        x_min, y_min = np.min(src_points, axis=0)
        x_max, y_max = np.max(src_points, axis=0)
        
        # 감지된 표지판의 크기
        detected_width = x_max - x_min
        detected_height = y_max - y_min
        detected_area = detected_width * detected_height
        
        # 템플릿 대비 스케일 비율 계산
        template_height, template_width = template_shape
        scale_ratio = min(detected_width / template_width, detected_height / template_height)
        
        return detected_area, detected_width, detected_height, scale_ratio

    def fnValidateSignDistance(self, area, width, height, scale_ratio):
        """
        📏 표지판 거리 검증 함수
        
        Returns:
            bool: True if 유효한 거리의 표지판, False if 너무 멀거나 가까운 표지판
        """
        
        # 크기 기반 필터링
        if area < self.MIN_SIGN_AREA or area > self.MAX_SIGN_AREA:
            return False
            
        if width < self.MIN_SIGN_WIDTH or height < self.MIN_SIGN_HEIGHT:
            return False
            
        # 스케일 비율 기반 필터링 (너무 작거나 큰 표지판 제외)
        if scale_ratio < 0.3 or scale_ratio > 2:  # 30% ~ 200% 범위만 허용
            return False
            
        return True
    def fnValidateSignDistance_red(self, area, width, height, scale_ratio):
        """
        📏 표지판 거리 검증 함수
        
        Returns:
            bool: True if 유효한 거리의 표지판, False if 너무 멀거나 가까운 표지판
        """
        
        # 크기 기반 필터링
        if area < self.MIN_SIGN_AREA or area > self.MAX_SIGN_AREA:
            return False
            
        if width < self.MIN_SIGN_WIDTH or height < self.MIN_SIGN_HEIGHT:
            return False
            
        # 스케일 비율 기반 필터링 (너무 작거나 큰 표지판 제외)
        if scale_ratio < 0.2 or scale_ratio > 3.5:  # 20% ~ 350% 범위만 허용
            return False
            
        return True

    def fnUpdateDetectionHistory(self, sign_type, detected):
        """
        📊 연속 프레임 검증을 위한 히스토리 업데이트
        
        Args:
            sign_type: 표지판 종류 ('stop', 'left', 'right')
            detected: 현재 프레임에서 감지 여부
            
        Returns:
            bool: 연속 감지 조건을 만족하는지 여부
        """
        
        if detected:
            self.detection_history[sign_type] = min(
                self.detection_history[sign_type] + 1, 
                self.MAX_HISTORY
            )
        else:
            self.detection_history[sign_type] = max(
                self.detection_history[sign_type] - 1, 
                0
            )
            
        return self.detection_history[sign_type] >= self.DETECTION_THRESHOLD

    def fnCheckDetectionInterval(self, sign_type):
        """
        ⏰ 중복 감지 방지를 위한 시간 간격 확인
        """
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        if current_time - self.last_detection_time[sign_type] > self.MIN_DETECTION_INTERVAL:
            self.last_detection_time[sign_type] = current_time
            return True
        return False

    def cbFindTrafficSign(self, image_msg):
        """
        🎯 교통 표지판 감지 메인 처리 함수 (거리 필터링 강화 버전)
        """
        
        # 프레임 스킵 처리
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        # 이미지 변환
        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image_input = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif self.sub_image_type == 'raw':
            cv_image_input = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')

        # 더 엄격한 임계값 설정
        MIN_MATCH_COUNT = 5     # 기존 5 → 10으로 증가 (더 엄격한 매칭)
        MIN_MSE_DECISION = 65000 # 기존 70000 → 40000으로 감소 (더 엄격한 품질 검증)

        # SIFT 특징점 추출
        kp1, des1 = self.sift.detectAndCompute(cv_image_input, None)

        # 각 템플릿과의 매칭
        matches_stop = self.flann.knnMatch(des1, self.des_stop, k=2)
        matches_left = self.flann.knnMatch(des1, self.des_left, k=2)
        matches_right = self.flann.knnMatch(des1, self.des_right, k=2)

        image_out_num = 1
        
        # 감지 플래그 초기화
        detected_signs = {
            'stop': False,
            'left': False,
            'right': False
        }

        # =============================================================
        # 🚦 정지 표지판 검출 및 검증 (거리 필터링 추가)
        # =============================================================
        good_stop = []
        for m, n in matches_stop:
            if m.distance < 0.7*n.distance:  # 기존 0.7 → 0.66으로 더 엄격하게
                good_stop.append(m)
                
        if len(good_stop) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_stop]).reshape(-1, 1, 2)
            dst_pts = np.float32([
                self.kp_stop[m.trainIdx].pt for m in good_stop
            ]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)  # 더 엄격한 RANSAC
            if M is not None:  # Homography 계산 성공 확인
                matches_stop = mask.ravel().tolist()

                # MSE 계산
                mse = self.fnCalcMSE(src_pts, dst_pts)
                
                # 표지판 크기 및 거리 검증
                area, width, height, scale_ratio = self.fnCalculateSignSize(
                    src_pts, dst_pts, self.template_sizes['stop']
                )
                
                # 모든 조건을 만족하는 경우에만 감지로 판단
                if (mse < MIN_MSE_DECISION and 
                    self.fnValidateSignDistance_red(area, width, height, scale_ratio)):
                    ###
                    
                    detected_signs['stop'] = True
                    self.get_logger().info(f'stop candidate - Area: {area:.0f}, Scale: {scale_ratio:.2f}, MSE: {mse:.0f}')

        # =============================================================
        # 👈 좌회전 표지판 검출 및 검증 (동일한 강화된 로직 적용)
        # =============================================================
        good_left = []
        for m, n in matches_left:
            if m.distance < 0.66*n.distance:
                good_left.append(m)
                
        if len(good_left) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_left]).reshape(-1, 1, 2)
            dst_pts = np.float32([
                self.kp_left[m.trainIdx].pt for m in good_left
            ]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            if M is not None:
                matches_left = mask.ravel().tolist()
                mse = self.fnCalcMSE(src_pts, dst_pts)
                
                area, width, height, scale_ratio = self.fnCalculateSignSize(
                    src_pts, dst_pts, self.template_sizes['left']
                )
                
                if (mse < MIN_MSE_DECISION and 
                    self.fnValidateSignDistance(area, width, height, scale_ratio)):
                    
                    detected_signs['left'] = True
                    self.get_logger().info(f'Left candidate - Area: {area:.0f}, Scale: {scale_ratio:.2f}, MSE: {mse:.0f}')
            else:
                matches_left = None
        else:
            matches_left = None

        # =============================================================
        # 👉 우회전 표지판 검출 및 검증 (동일한 강화된 로직 적용)
        # =============================================================
        good_right = []
        for m, n in matches_right:
            if m.distance < 0.66*n.distance:
                good_right.append(m)
                
        if len(good_right) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_right]).reshape(-1, 1, 2)
            dst_pts = np.float32([
                self.kp_right[m.trainIdx].pt for m in good_right
            ]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            if M is not None:
                matches_right = mask.ravel().tolist()
                mse = self.fnCalcMSE(src_pts, dst_pts)
                
                area, width, height, scale_ratio = self.fnCalculateSignSize(
                    src_pts, dst_pts, self.template_sizes['right']
                )
                
                if (mse < MIN_MSE_DECISION and 
                    self.fnValidateSignDistance(area, width, height, scale_ratio)):
                    
                    detected_signs['right'] = True
                    self.get_logger().info(f'Right candidate - Area: {area:.0f}, Scale: {scale_ratio:.2f}, MSE: {mse:.0f}')
            else:
                matches_right = None
        else:
            matches_right = None

        # =============================================================
        # 📊 연속 프레임 검증 및 최종 감지 결과 발행
        # =============================================================
        
        # 교차로 표지판 최종 검증
        if self.fnUpdateDetectionHistory('stop', detected_signs['stop']):
            if self.fnCheckDetectionInterval('stop'):
                msg_sign = UInt8()
                msg_sign.data = self.TrafficSign.stop.value
                self.pub_traffic_sign.publish(msg_sign)
                
                msg_stop = Bool()
                msg_stop.data = True
                self.pub_stop_detected.publish(msg_stop)
                
                self.get_logger().info('✅ CONFIRMED: stop sign detected!')
                image_out_num = 2

        # 좌회전 표지판 최종 검증
        if self.fnUpdateDetectionHistory('left', detected_signs['left']):
            if self.fnCheckDetectionInterval('left'):
                msg_sign = UInt8()
                msg_sign.data = self.TrafficSign.left.value
                self.pub_traffic_sign.publish(msg_sign)
                
                msg_left = Bool()
                msg_left.data = True
                self.pub_left_detected.publish(msg_left)
                
                self.get_logger().info('✅ CONFIRMED: Left sign detected!')
                image_out_num = 3

        # 우회전 표지판 최종 검증
        if self.fnUpdateDetectionHistory('right', detected_signs['right']):
            if self.fnCheckDetectionInterval('right'):
                msg_sign = UInt8()
                msg_sign.data = self.TrafficSign.right.value
                self.pub_traffic_sign.publish(msg_sign)
                
                msg_right = Bool()
                msg_right.data = True
                self.pub_right_detected.publish(msg_right)
                
                self.get_logger().info('✅ CONFIRMED: Right sign detected!')
                image_out_num = 4

        # =============================================================
        # 🖼️ 결과 이미지 생성 및 발행
        # =============================================================
        
        if image_out_num == 1:
            # 표지판이 감지되지 않은 경우
            if self.pub_image_type == 'compressed':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(cv_image_input, 'jpg')
                )
            elif self.pub_image_type == 'raw':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_imgmsg(cv_image_input, 'bgr8')
                )
                
        elif image_out_num == 2:
            # 교차로 표지판 감지 시각화
            draw_params_stop = {
                'matchColor': (0, 255, 0),          # 초록색으로 변경 (확인된 감지)
                'singlePointColor': None,
                'matchesMask': matches_stop,
                'flags': 2
            }
            
            final_stop = cv2.drawMatches(
                cv_image_input, kp1, self.img_stop, self.kp_stop,
                good_stop, None, **draw_params_stop
            )

            if self.pub_image_type == 'compressed':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(final_stop, 'jpg')
                )
            elif self.pub_image_type == 'raw':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_imgmsg(final_stop, 'bgr8')
                )
                
        elif image_out_num == 3:
            # 좌회전 표지판 감지 시각화
            draw_params_left = {
                'matchColor': (0, 255, 0),
                'singlePointColor': None,
                'matchesMask': matches_left,
                'flags': 2
            }

            final_left = cv2.drawMatches(
                cv_image_input, kp1, self.img_left, self.kp_left,
                good_left, None, **draw_params_left
            )

            if self.pub_image_type == 'compressed':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(final_left, 'jpg')
                )
            elif self.pub_image_type == 'raw':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_imgmsg(final_left, 'bgr8')
                )
                
        elif image_out_num == 4:
            # 우회전 표지판 감지 시각화
            draw_params_right = {
                'matchColor': (0, 255, 0),
                'singlePointColor': None,
                'matchesMask': matches_right,
                'flags': 2
            }
            
            final_right = cv2.drawMatches(
                cv_image_input, kp1, self.img_right, self.kp_right,
                good_right, None, **draw_params_right
            )

            if self.pub_image_type == 'compressed':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(final_right, 'jpg')
                )
            elif self.pub_image_type == 'raw':
                self.pub_image_traffic_sign.publish(
                    self.cvBridge.cv2_to_imgmsg(final_right, 'bgr8')
                )

def main(args=None):
    """
    🚀 프로그램 메인 진입점
    """
    rclpy.init(args=args)
    node = DetectSign()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()