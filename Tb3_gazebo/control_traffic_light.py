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
# Author: Your Name
# Purpose: Traffic light control module for autonomous driving
# 
# ============================================================================
# 이 프로그램의 주요 기능:
# 1. 카메라나 센서로 감지된 신호등 정보를 받아서 분석
# 2. 신호등 색상(빨강, 노랑, 초록)에 따라 로봇의 움직임을 제어
# 3. 빨간불/노란불: 정지, 초록불: 진행 허용
# 4. 감지 신뢰도를 확인하여 잘못된 판단 방지
# 5. 안전을 위한 다양한 보호 기능 제공
# ============================================================================

import time
from geometry_msgs.msg import Twist  # 로봇 움직임 명령 메시지 (속도, 회전)
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool  # 참/거짓 메시지
from std_msgs.msg import UInt8  # 8비트 정수 메시지 (신호등 상태용)


class ControlTraffic(Node):
    """
    신호등 제어 클래스
    - 신호등을 감지하고 그에 따라 로봇의 움직임을 제어하는 ROS2 노드
    - 안전을 최우선으로 하여 여러 보호 장치를 포함
    """

    def __init__(self):
        super().__init__('control_traffic')  # 노드 이름: 'control_traffic'

        # =============================================================
        # 1. 구독자(Subscriber) 설정 - 다른 노드에서 정보를 받기 위함
        # =============================================================
        
        # 신호등 상태 구독자 (메인 신호등 정보)
        # 다른 노드에서 감지한 신호등 색상 정보를 받음
        self.sub_traffic_light = self.create_subscription(
            UInt8,  # 메시지 타입: 8비트 정수 (0=없음, 1=빨강, 2=노랑, 3=초록)
            '/detect/traffic_light',  # 토픽 이름
            self.callback_traffic_light,  # 메시지 받았을 때 실행할 함수
            1  # 큐 크기
        )

        # 신뢰도 정보 구독자들 - 각 색상별로 얼마나 확실한지 받음 (0~100%)
        self.sub_red_reliability = self.create_subscription(
            UInt8,
            '/detect/red_light_reliability',  # 빨간불 신뢰도
            self.callback_red_reliability,
            1
        )
        self.sub_yellow_reliability = self.create_subscription(
            UInt8,
            '/detect/yellow_light_reliability',  # 노란불 신뢰도
            self.callback_yellow_reliability,
            1
        )
        self.sub_green_reliability = self.create_subscription(
            UInt8,
            '/detect/green_light_reliability',  # 초록불 신뢰도
            self.callback_green_reliability,
            1
        )

        # 기존 움직임 명령 구독 (다른 제어기에서 오는 명령)
        # 예: 장애물 회피, 경로 추종 등의 제어기에서 오는 속도 명령
        self.sub_cmd_vel_input = self.create_subscription(
            Twist,  # 선속도(앞뒤)와 각속도(회전) 포함
            '/control/cmd_vel',
            self.callback_cmd_vel_input,
            1
        )

        # =============================================================
        # 2. 발행자(Publisher) 설정 - 다른 노드에 정보를 보내기 위함
        # =============================================================
        
        # 최종 움직임 명령 발행자 (신호등 제어가 적용된 최종 명령)
        # 실제 로봇 모터에 전달되는 최종 속도 명령
        self.pub_cmd_vel_final = self.create_publisher(
            Twist,
            '/cmd_vel',  # 로봇이 실제로 읽는 속도 명령 토픽
            1
        )

        # 신호등 제어 상태 발행자 (다른 노드에서 참조용)
        # 현재 신호등 때문에 정지 중인지 알려주는 용도
        self.pub_traffic_override = self.create_publisher(
            Bool,
            '/traffic_light_override',
            1
        )

        # =============================================================
        # 3. 신호등 상태 상수 정의
        # =============================================================
        self.TRAFFIC_LIGHT_NONE = 0    # 신호등 없음
        self.TRAFFIC_LIGHT_RED = 1     # 빨간불
        self.TRAFFIC_LIGHT_YELLOW = 2  # 노란불
        self.TRAFFIC_LIGHT_GREEN = 3   # 초록불
        
        # =============================================================
        # 4. 현재 상태를 저장하는 변수들
        # =============================================================
        self.current_traffic_state = self.TRAFFIC_LIGHT_NONE  # 현재 신호등 상태
        self.traffic_override_active = False  # 신호등 때문에 정지 중인가?
        
        # =============================================================
        # 5. 신뢰도 관련 변수들 - 잘못된 감지를 방지하기 위함
        # =============================================================
        self.red_reliability = 100     # 빨간불 감지 신뢰도 (0~100%)
        self.yellow_reliability = 100  # 노란불 감지 신뢰도
        self.green_reliability = 100   # 초록불 감지 신뢰도
        self.min_reliability_threshold = 70  # 최소 신뢰도 임계값 (70% 이상이어야 믿음)
        self.reliability_override_count = 0  # 신뢰도 부족으로 무시한 횟수
        
        # =============================================================
        # 6. 상태 지속성을 위한 변수들 - 깜빡임이나 오감지 방지
        # =============================================================
        self.none_detection_start = None  # 신호등이 안 보이기 시작한 시간
        self.NONE_TIMEOUT = 2.0          # 신호등이 안 보여도 이전 상태 유지할 시간 (초)
        self.RED_PERSISTENCE_TIME = 3.0  # 빨간불이 안 보여도 정지 상태 유지할 시간 (초)
        
        # =============================================================
        # 7. 로깅 및 안전 관련 변수들
        # =============================================================
        self.last_log_time = 0.0      # 마지막 로그 출력 시간
        self.log_interval = 2.0       # 로그 출력 간격 (초) - 너무 자주 출력하지 않기 위함
        self.last_cmd_vel_time = time.time()  # 마지막으로 속도 명령을 받은 시간
        self.cmd_vel_timeout = 1.0    # 속도 명령 타임아웃 (초)
        
        # =============================================================
        # 8. 기본 제어 관련 변수들 - 다른 제어기가 없을 때 사용
        # =============================================================
        self.MAX_VEL = 0.1           # 최대 속도 (m/s)
        self.basic_control_active = True  # 기본 제어 활성화 여부
        
        # =============================================================
        # 9. 타이머 설정 - 주기적으로 실행할 함수들
        # =============================================================
        # 안전 체크 타이머 (0.1초마다 실행)
        self.safety_timer = self.create_timer(0.1, self.safety_check)
        
        # 기본 제어 타이머 (0.1초마다 실행)
        self.basic_control_timer = self.create_timer(0.1, self.basic_traffic_control)

        self.get_logger().info('Traffic Light Control Module with Reliability Check Initialized')

    # =============================================================
    # 신뢰도 정보 업데이트 함수들
    # =============================================================
    
    def callback_red_reliability(self, msg):
        """빨간불 신뢰도 업데이트 (0~100% 값)"""
        self.red_reliability = msg.data

    def callback_yellow_reliability(self, msg):
        """노란불 신뢰도 업데이트 (0~100% 값)"""
        self.yellow_reliability = msg.data

    def callback_green_reliability(self, msg):
        """초록불 신뢰도 업데이트 (0~100% 값)"""
        self.green_reliability = msg.data

    def callback_traffic_light(self, traffic_light_msg):
        """
        신호등 상태 업데이트 메인 함수
        - 신뢰도를 확인한 후 믿을 만하면 상태 업데이트
        - 신뢰도가 낮으면 이전 상태 유지 (안전을 위함)
        """
        raw_state = traffic_light_msg.data  # 받은 신호등 상태 (0,1,2,3)
        
        # 현재 감지된 상태의 신뢰도가 충분한지 확인
        is_reliable = self.check_state_reliability(raw_state)
        
        if is_reliable:
            # 신뢰도가 충분하면 정상 처리
            processed_state = self.process_traffic_state_with_persistence(raw_state)
            
            # 상태가 실제로 바뀐 경우에만 업데이트
            if processed_state != self.current_traffic_state:
                self.current_traffic_state = processed_state
                self.update_traffic_override()  # 제어 상태 업데이트
                self.log_traffic_state_change()  # 로그 출력
                self.reliability_override_count = 0  # 카운터 리셋
        else:
            # 신뢰도가 낮으면 이전 상태 유지
            self.reliability_override_count += 1
            
            # 주기적으로 경고 로그 출력 (너무 자주 출력하지 않음)
            if self.reliability_override_count % 20 == 1:
                self.get_logger().warn(f'Low reliability: {self.get_state_name(raw_state)} - maintaining current state')

    def process_traffic_state_with_persistence(self, raw_state):
        """
        상태 지속성을 적용한 신호등 상태 처리
        - 신호등이 갑자기 안 보이거나 깜빡거리는 경우 대응
        - 특히 빨간불에서는 더 오래 정지 상태 유지 (안전)
        """
        current_time = time.time()
        
        # 확실한 신호 감지된 경우 (빨강, 노랑, 초록)
        if raw_state in [self.TRAFFIC_LIGHT_RED, self.TRAFFIC_LIGHT_YELLOW, self.TRAFFIC_LIGHT_GREEN]:
            self.none_detection_start = None  # NONE 감지 시간 리셋
            return raw_state
        
        # 신호등이 안 보이는 경우 (NONE) 처리
        else:
            current_state = self.current_traffic_state
            
            # NONE 감지 시작 시간 기록
            if self.none_detection_start is None:
                self.none_detection_start = current_time
            
            time_since_none = current_time - self.none_detection_start
            
            # 빨간불에서 NONE이 된 경우 - 특히 주의 (안전상 더 오래 정지)
            if current_state == self.TRAFFIC_LIGHT_RED:
                if time_since_none < self.RED_PERSISTENCE_TIME:
                    return self.TRAFFIC_LIGHT_RED  # 계속 빨간불로 유지
                else:
                    return self.TRAFFIC_LIGHT_NONE  # 충분히 기다렸으면 NONE으로
            
            # 노란불이나 초록불에서 NONE이 된 경우
            elif current_state in [self.TRAFFIC_LIGHT_YELLOW, self.TRAFFIC_LIGHT_GREEN]:
                if time_since_none < self.NONE_TIMEOUT:
                    return current_state  # 잠시 이전 상태 유지
                else:
                    return self.TRAFFIC_LIGHT_NONE  # 시간 지나면 NONE으로
            
            # 이미 NONE 상태인 경우
            else:
                return self.TRAFFIC_LIGHT_NONE

    def check_state_reliability(self, state):
        """
        현재 상태의 신뢰도가 충분한지 확인
        - 각 색상별 신뢰도가 임계값(70%) 이상인지 체크
        """
        current_reliability = self.get_current_state_reliability(state)
        return current_reliability >= self.min_reliability_threshold

    def get_current_state_reliability(self, state):
        """
        현재 상태의 신뢰도 값 반환
        - 감지된 색상에 해당하는 신뢰도 값을 가져옴
        """
        if state == self.TRAFFIC_LIGHT_RED:
            return self.red_reliability
        elif state == self.TRAFFIC_LIGHT_YELLOW:
            return self.yellow_reliability
        elif state == self.TRAFFIC_LIGHT_GREEN:
            return self.green_reliability
        else:  # NONE인 경우
            return 100  # NONE은 항상 신뢰도 100%로 간주

    def get_state_name(self, state):
        """상태 코드를 사람이 읽기 쉬운 이름으로 변환"""
        state_names = {
            self.TRAFFIC_LIGHT_NONE: 'NONE',
            self.TRAFFIC_LIGHT_RED: 'RED',
            self.TRAFFIC_LIGHT_YELLOW: 'YELLOW',
            self.TRAFFIC_LIGHT_GREEN: 'GREEN'
        }
        return state_names.get(state, 'UNKNOWN')

    def update_traffic_override(self):
        """
        신호등 제어 오버라이드 상태 업데이트
        - 빨간불/노란불일 때 오버라이드 활성화 (정지 명령)
        - 초록불/없음일 때 오버라이드 비활성화 (정상 주행 허용)
        """
        # 빨간불/노란불일 때 오버라이드 활성화
        if (self.current_traffic_state == self.TRAFFIC_LIGHT_RED or 
            self.current_traffic_state == self.TRAFFIC_LIGHT_YELLOW):
            self.traffic_override_active = True
        else:
            self.traffic_override_active = False
        
        # 다른 노드들에게 현재 오버라이드 상태 알림
        override_msg = Bool()
        override_msg.data = self.traffic_override_active
        self.pub_traffic_override.publish(override_msg)

    def callback_cmd_vel_input(self, cmd_vel_msg):
        """
        다른 제어기에서 오는 속도 명령 처리
        - 장애물 회피, 경로 추종 등의 제어기에서 오는 명령을 받음
        - 신호등 상태를 고려하여 최종 명령 결정
        """
        self.last_cmd_vel_time = time.time()  # 명령 받은 시간 기록
        
        # 외부 명령이 있으면 기본 제어 비활성화
        self.basic_control_active = False
        
        # 신호등 상태에 따라 최종 속도 명령 결정
        final_cmd_vel = self.apply_traffic_control(cmd_vel_msg)
        self.pub_cmd_vel_final.publish(final_cmd_vel)

    def apply_traffic_control(self, input_cmd_vel):
        """
        신호등 상태에 따른 제어 적용
        - 빨간불/노란불: 강제 정지
        - 초록불/없음: 입력 명령 그대로 통과
        """
        final_cmd_vel = Twist()
        
        if (self.current_traffic_state == self.TRAFFIC_LIGHT_RED or 
            self.current_traffic_state == self.TRAFFIC_LIGHT_YELLOW):
            # 빨간불/노란불: 완전 정지 (안전 최우선)
            final_cmd_vel.linear.x = 0.0   # 전진 속도 0
            final_cmd_vel.angular.z = 0.0  # 회전 속도 0
            
        elif self.current_traffic_state == self.TRAFFIC_LIGHT_GREEN:
            # 초록불: 입력받은 속도 명령 그대로 통과
            final_cmd_vel = input_cmd_vel
            
        else:  # TRAFFIC_LIGHT_NONE (신호등 없음)
            # 신호등 없음: 입력받은 속도 명령 그대로 통과
            final_cmd_vel = input_cmd_vel
        
        return final_cmd_vel

    def basic_traffic_control(self):
        """
        기본 신호등 제어 - 다른 제어기에서 명령이 없을 때 사용
        - 신호등 상태에 따라 기본적인 동작 수행
        - 초록불: 천천히 직진, 빨간불: 정지, 없음: 매우 천천히 직진
        """
        current_time = time.time()
        time_since_last_cmd = current_time - self.last_cmd_vel_time
        
        # 외부 속도 명령이 일정 시간 없으면 기본 제어 활성화
        if time_since_last_cmd > 0.5:  # 0.5초 이상 명령이 없으면
            self.basic_control_active = True
        
        # 기본 제어가 활성화된 경우에만 실행
        if not self.basic_control_active:
            return

        twist = Twist()
        
        # 신호등 상태에 따른 기본 제어
        if (self.current_traffic_state == self.TRAFFIC_LIGHT_RED or 
            self.current_traffic_state == self.TRAFFIC_LIGHT_YELLOW):
            # 빨간불/노란불: 완전 정지
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            
        elif self.current_traffic_state == self.TRAFFIC_LIGHT_GREEN:
            # 초록불: 천천히 직진 (안전을 위해 절반 속도)
            twist.linear.x = self.MAX_VEL * 0.5
            twist.angular.z = 0.0  # 직진만
            
        else:  # TRAFFIC_LIGHT_NONE
            # 신호등 없음: 매우 천천히 직진 (더욱 안전하게)
            twist.linear.x = self.MAX_VEL * 0.3
            twist.angular.z = 0.0  # 직진만
        
        # 기본 제어 속도 명령 발행
        self.pub_cmd_vel_final.publish(twist)

    def safety_check(self):
        """
        주기적 안전 체크 (0.1초마다 실행)
        - 외부 명령이 끊어진 경우 안전 정지
        - 시스템 전체의 안전성 확보
        """
        current_time = time.time()
        
        # 외부 속도 명령 타임아웃 체크 (기본 제어가 비활성화된 경우에만)
        if current_time - self.last_cmd_vel_time > self.cmd_vel_timeout and not self.basic_control_active:
            # 타임아웃 시 즉시 안전 정지
            safety_stop = Twist()  # 모든 속도를 0으로 설정
            self.pub_cmd_vel_final.publish(safety_stop)
            
            # 경고 로그 출력 (너무 자주 출력하지 않음)
            if self.should_log():
                self.get_logger().warn('cmd_vel timeout - Safety stop activated')

    def log_traffic_state_change(self):
        """
        신호등 상태 변경 로그 출력
        - 현재 신호등 상태와 제어 상태를 명확히 표시
        """
        state_name = self.get_state_name(self.current_traffic_state)
        override_status = "STOP" if self.traffic_override_active else "GO"
        
        # 이모지를 사용하여 직관적으로 표시
        self.get_logger().info(f'🚦 Traffic Light: {state_name} - {override_status}')

    def should_log(self):
        """
        로그 출력 빈도 제한
        - 너무 자주 로그가 출력되는 것을 방지
        - 지정된 간격(2초)마다만 로그 허용
        """
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self.last_log_time = current_time
            return True
        return False

    def shut_down(self):
        """
        노드 종료 시 안전 정지
        - 프로그램이 종료될 때 로봇을 안전하게 정지
        """
        self.get_logger().info('Traffic Control shutting down')
        stop_twist = Twist()  # 모든 속도를 0으로 설정
        self.pub_cmd_vel_final.publish(stop_twist)


def main(args=None):
    """
    메인 함수 - 프로그램 시작점
    - ROS2 초기화 및 노드 실행
    - 키보드 인터럽트(Ctrl+C) 처리
    - 종료 시 안전 정리
    """
    rclpy.init(args=args)  # ROS2 초기화
    node = ControlTraffic()  # 신호등 제어 노드 생성
    
    try:
        rclpy.spin(node)  # 노드 실행 (메시지 수신 대기)
    except KeyboardInterrupt:
        pass  # Ctrl+C로 종료 시 정상 처리
    finally:
        node.shut_down()      # 안전 정지
        node.destroy_node()   # 노드 정리
        rclpy.shutdown()      # ROS2 종료


if __name__ == '__main__':
    main()