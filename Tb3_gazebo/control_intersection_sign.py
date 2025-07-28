#!/usr/bin/env python3
#
# 교통 표지판 인식에 따른 제어 노드
# 
# ============================================================================
# 이 프로그램의 주요 기능:
# 1. 카메라로 감지된 교통 표지판 정보를 받아서 분석
# 2. 교차로 표지판: 일시정지 후 천천히 통과
# 3. 좌회전 표지판: 좌회전 동작 수행
# 4. 우회전 표지판: 우회전 동작 수행
# 5. 상태 기반 제어로 안전하고 순차적인 동작 보장
# 6. 중복 처리 방지를 위한 쿨다운 시스템
# ============================================================================

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool          # 참/거짓 메시지 (표지판 감지 여부)
from geometry_msgs.msg import Twist    # 로봇 움직임 명령 메시지
from enum import Enum                  # 상태 정의를 위한 열거형
import time

class ControlState(Enum):
    """
    로봇의 제어 상태 정의
    - 상태 기반 제어(State Machine)를 위한 열거형
    - 각 상태별로 다른 동작을 수행
    """
    NORMAL = 0                    # 일반 상태 (표지판 감지 대기)
    PROCESSING_stop = 1   # 교차로 처리 중 (정지 후 천천히 통과)
    PROCESSING_LEFT_TURN = 2      # 좌회전 처리 중
    PROCESSING_RIGHT_TURN = 3     # 우회전 처리 중
    WAITING = 4                   # 대기 상태 (동작 완료 후 정상 상태 복귀 전)

class ControlstopSign(Node):
    """
    교통 표지판 제어 클래스
    - 교통 표지판을 인식하고 그에 따라 로봇의 움직임을 제어하는 ROS2 노드
    - 상태 기반 제어로 안전하고 체계적인 동작 수행
    """

    def __init__(self):
        super().__init__('control_stop_sign')  # 노드 이름 설정
        
        # =============================================================
        # 1. 구독자(Subscriber) 설정 - 표지판 감지 정보 수신
        # =============================================================
        
        # 교차로 표지판 감지 구독자
        # 다른 노드에서 교차로 표지판을 감지했는지 정보를 받음
        self.sub_stop = self.create_subscription(
            Bool,                           # 메시지 타입: 참/거짓
            '/detect/stop_sign',    # 토픽 이름
            self.cb_stop_detected,  # 메시지 받았을 때 실행할 함수
            10                             # 큐 크기
        )
        
        # 좌회전 표지판 감지 구독자
        self.sub_left = self.create_subscription(
            Bool,
            '/detect/left_sign',
            self.cb_left_detected,
            10
        )
        
        # 우회전 표지판 감지 구독자
        self.sub_right = self.create_subscription(
            Bool,
            '/detect/right_sign',
            self.cb_right_detected,
            10
        )
        
        # =============================================================
        # 2. 발행자(Publisher) 설정 - 로봇 제어 명령 송신
        # =============================================================
        
        # 로봇 움직임 제어 명령 발행자
        # 실제 로봇 모터에 전달되는 속도 명령을 발행
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # =============================================================
        # 3. 상태 관리 변수들
        # =============================================================
        self.control_state = ControlState.NORMAL  # 현재 제어 상태
        self.last_detection_time = 0.0           # 마지막 표지판 감지 시간
        self.detection_cooldown = 3.0            # 표지판 감지 쿨다운 시간 (초)
        
        # =============================================================
        # 4. 제어 파라미터들 - 로봇 동작의 속도와 시간 설정
        # =============================================================
        self.linear_speed = 0.2     # 기본 직진 속도 (m/s)
        self.angular_speed = 0.5    # 기본 회전 속도 (rad/s)
        self.stop_duration = 2.0    # 교차로에서 정지 시간 (초)
        self.turn_duration = 2.0    # 좌/우회전 동작 시간 (초)
        
        # =============================================================
        # 5. 타이머 및 시간 관리
        # =============================================================
        # 메인 제어 루프 타이머 (0.1초마다 실행 = 10Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.action_start_time = 0.0  # 현재 동작 시작 시간
        
        # 초기화 완료 메시지
        self.get_logger().info('ControlstopSign Node Initialized')
        self.get_logger().info('Waiting for traffic sign detection...')

    # =============================================================
    # 표지판 감지 콜백 함수들
    # =============================================================
    
    def cb_stop_detected(self, msg):
        """
        교차로 표지판 인식 콜백 함수
        - 교차로 표지판이 감지되면 호출
        - 안전을 위해 일시정지 후 천천히 통과하는 동작 시작
        """
        if msg.data and self.can_process_detection():
            self.get_logger().info('🚦 stop sign detected! Starting stop control sequence')
            self.control_state = ControlState.PROCESSING_stop
            self.action_start_time = self.get_clock().now().nanoseconds / 1e9  # 현재 시간 기록
            self.last_detection_time = time.time()  # 감지 시간 기록 (쿨다운용)
            self.stop_robot()  # 즉시 정지

    def cb_left_detected(self, msg):
        """
        좌회전 표지판 인식 콜백 함수
        - 좌회전 표지판이 감지되면 호출
        - 좌회전 동작 시퀀스 시작
        """
        if msg.data and self.can_process_detection():
            self.get_logger().info('👈 Left turn sign detected! Starting left turn sequence')
            self.control_state = ControlState.PROCESSING_LEFT_TURN
            self.action_start_time = self.get_clock().now().nanoseconds / 1e9
            self.last_detection_time = time.time()

    def cb_right_detected(self, msg):
        """
        우회전 표지판 인식 콜백 함수
        - 우회전 표지판이 감지되면 호출
        - 우회전 동작 시퀀스 시작
        """
        if msg.data and self.can_process_detection():
            self.get_logger().info('👉 Right turn sign detected! Starting right turn sequence')
            self.control_state = ControlState.PROCESSING_RIGHT_TURN
            self.action_start_time = self.get_clock().now().nanoseconds / 1e9
            self.last_detection_time = time.time()

    def can_process_detection(self):
        """
        새로운 표지판 감지를 처리할 수 있는지 확인
        - 중복 처리 방지를 위한 안전 장치
        - 현재 다른 동작 중이거나 쿨다운 중이면 무시
        
        Returns:
            bool: 처리 가능하면 True, 불가능하면 False
        """
        current_time = time.time()
        
        # 현재 다른 동작을 처리 중이면 새로운 감지 무시
        if self.control_state != ControlState.NORMAL:
            return False
            
        # 쿨다운 시간이 지나지 않았으면 무시 (연속 감지 방지)
        if current_time - self.last_detection_time < self.detection_cooldown:
            return False
            
        return True

    def control_loop(self):
        """
        메인 제어 루프 (0.1초마다 실행)
        - 현재 상태에 따라 적절한 제어 함수 호출
        - 시간 기반 제어로 정확한 동작 수행
        """
        current_time = self.get_clock().now().nanoseconds / 1e9  # 현재 시간 (초)
        elapsed_time = current_time - self.action_start_time     # 동작 시작 후 경과 시간
        
        # 현재 상태에 따른 제어 함수 호출
        if self.control_state == ControlState.PROCESSING_stop:
            self.handle_stop_control(elapsed_time)
        elif self.control_state == ControlState.PROCESSING_LEFT_TURN:
            self.handle_left_turn_control(elapsed_time)
        elif self.control_state == ControlState.PROCESSING_RIGHT_TURN:
            self.handle_right_turn_control(elapsed_time)
        elif self.control_state == ControlState.WAITING:
            self.handle_waiting_state(elapsed_time)

    # =============================================================
    # 상태별 제어 처리 함수들
    # =============================================================

    def handle_stop_control(self, elapsed_time):
        """
        교차로 제어 처리 함수
        - 1단계: 지정된 시간 동안 완전 정지
        - 2단계: 천천히 전진하며 교차로 통과
        - 3단계: 정상 상태로 복귀
        
        Args:
            elapsed_time (float): 교차로 제어 시작 후 경과 시간 (초)
        """
        if elapsed_time < self.stop_duration:
            # 1단계: 정지 상태 유지 (안전 확인)
            self.stop_robot()
            if elapsed_time < 0.1:  # 첫 번째 제어 루프에서만 로그 출력
                self.get_logger().info(f'🛑 Stopping at stop for {self.stop_duration} seconds')
                
        elif elapsed_time < self.stop_duration + 1.0:
            # 2단계: 천천히 직진하며 교차로 통과
            self.move_forward_slow()
            if elapsed_time < self.stop_duration + 0.1:
                self.get_logger().info('🚗 Proceeding through stop slowly')
                
        else:
            # 3단계: 교차로 통과 완료, 정상 상태로 복귀
            self.get_logger().info('✅ stop sequence completed')
            self.control_state = ControlState.NORMAL
            self.stop_robot()

    def handle_left_turn_control(self, elapsed_time):
        """
        좌회전 제어 처리 함수
        - 지정된 시간 동안 좌회전 동작 수행
        - 완료 후 대기 상태로 전환
        
        Args:
            elapsed_time (float): 좌회전 시작 후 경과 시간 (초)
        """
        if elapsed_time < self.turn_duration:
            # 좌회전 동작 수행
            self.turn_left()
            if elapsed_time < 0.1:
                self.get_logger().info(f'🔄 Turning left for {self.turn_duration} seconds')
        else:
            # 좌회전 완료, 대기 상태로 전환
            self.get_logger().info('✅ Left turn completed')
            self.control_state = ControlState.WAITING
            self.action_start_time = self.get_clock().now().nanoseconds / 1e9
            self.stop_robot()

    def handle_right_turn_control(self, elapsed_time):
        """
        우회전 제어 처리 함수
        - 지정된 시간 동안 우회전 동작 수행
        - 완료 후 대기 상태로 전환
        
        Args:
            elapsed_time (float): 우회전 시작 후 경과 시간 (초)
        """
        if elapsed_time < self.turn_duration:
            # 우회전 동작 수행
            self.turn_right()
            if elapsed_time < 0.1:
                self.get_logger().info(f'🔄 Turning right for {self.turn_duration} seconds')
        else:
            # 우회전 완료, 대기 상태로 전환
            self.get_logger().info('✅ Right turn completed')
            self.control_state = ControlState.WAITING
            self.action_start_time = self.get_clock().now().nanoseconds / 1e9
            self.stop_robot()

    def handle_waiting_state(self, elapsed_time):
        """
        대기 상태 처리 함수
        - 회전 동작 완료 후 잠시 대기
        - 안정화 시간을 거쳐 정상 상태로 복귀
        
        Args:
            elapsed_time (float): 대기 시작 후 경과 시간 (초)
        """
        if elapsed_time < 1.0:  # 1초 대기
            self.stop_robot()  # 대기 중 정지 상태 유지
        else:
            # 대기 완료, 정상 상태로 복귀
            self.get_logger().info('🔄 Returning to normal state')
            self.control_state = ControlState.NORMAL

    # =============================================================
    # 기본 로봇 제어 함수들
    # =============================================================

    def stop_robot(self):
        """
        로봇 완전 정지
        - 모든 속도(직진, 회전)를 0으로 설정
        """
        twist = Twist()
        twist.linear.x = 0.0   # 직진 속도 0
        twist.angular.z = 0.0  # 회전 속도 0
        self.pub_cmd_vel.publish(twist)

    def move_forward_slow(self):
        """
        천천히 전진
        - 교차로 통과 시 안전을 위해 느린 속도로 직진
        """
        twist = Twist()
        twist.linear.x = self.linear_speed * 0.5  # 기본 속도의 절반
        twist.angular.z = 0.0                     # 회전 없이 직진만
        self.pub_cmd_vel.publish(twist)

    def turn_left(self):
        """
        좌회전 동작
        - 느린 속도로 전진하면서 좌회전 수행
        - 제자리 회전이 아닌 곡선 회전으로 자연스러운 동작
        """
        twist = Twist()
        twist.linear.x = self.linear_speed * 0.3  # 느린 속도로 전진
        twist.angular.z = self.angular_speed      # 양수: 좌회전 (반시계방향)
        self.pub_cmd_vel.publish(twist)

    def turn_right(self):
        """
        우회전 동작
        - 느린 속도로 전진하면서 우회전 수행
        - 제자리 회전이 아닌 곡선 회전으로 자연스러운 동작
        """
        twist = Twist()
        twist.linear.x = self.linear_speed * 0.3   # 느린 속도로 전진
        twist.angular.z = -self.angular_speed      # 음수: 우회전 (시계방향)
        self.pub_cmd_vel.publish(twist)

    def get_current_state_string(self):
        """
        현재 상태를 사람이 읽기 쉬운 문자열로 반환
        - 디버깅이나 모니터링 목적으로 사용
        
        Returns:
            str: 현재 상태를 나타내는 문자열
        """
        state_strings = {
            ControlState.NORMAL: "Normal",
            ControlState.PROCESSING_stop: "Processing stop",
            ControlState.PROCESSING_LEFT_TURN: "Processing Left Turn",
            ControlState.PROCESSING_RIGHT_TURN: "Processing Right Turn",
            ControlState.WAITING: "Waiting"
        }
        return state_strings.get(self.control_state, "Unknown")

def main(args=None):
    """
    메인 함수 - 프로그램 시작점
    - ROS2 초기화 및 노드 실행
    - 예외 처리 및 안전한 종료
    """
    rclpy.init(args=args)  # ROS2 초기화
    
    try:
        node = ControlstopSign()  # 교통 표지판 제어 노드 생성
        rclpy.spin(node)                  # 노드 실행 (메시지 수신 대기)
    except KeyboardInterrupt:
        pass  # Ctrl+C로 종료 시 정상 처리
    finally:
        # 종료 시 안전 처리
        if 'node' in locals():
            node.stop_robot()      # 로봇 정지
            node.destroy_node()    # 노드 정리
        rclpy.shutdown()           # ROS2 종료

if __name__ == '__main__':
    main()