#!/usr/bin/env python3
#b/src/turtlebot3_autorace_detect/detect_lane_hybrid.py
import rclpy
from rclpy.node            import Node
from sensor_msgs.msg       import Image, CompressedImage
from std_msgs.msg          import Float64, UInt8
from cv_bridge             import CvBridge
from geometry_msgs.msg     import PointStamped
import cv2
import numpy as np
from collections           import deque

class DetectLane(Node):
    def __init__(self):
        super().__init__('detect_lane')

        # === 1) ROS 파라미터 (detect_lane.py와 동일 네임스페이스) :contentReference[oaicite:0]{index=0} ===
        self.declare_parameters(
            namespace='',
            parameters=[
                # White HSV
                ('detect.lane.white.hue_l',           0),
                ('detect.lane.white.hue_h',         179),
                ('detect.lane.white.saturation_l',    0),
                ('detect.lane.white.saturation_h',   70),
                ('detect.lane.white.lightness_l',   105),
                ('detect.lane.white.lightness_h',   255),
                # Yellow HSV
                ('detect.lane.yellow.hue_l',         10),
                ('detect.lane.yellow.hue_h',        127),
                ('detect.lane.yellow.saturation_l',  70),
                ('detect.lane.yellow.saturation_h', 255),
                ('detect.lane.yellow.lightness_l',   95),
                ('detect.lane.yellow.lightness_h',  255),
                # Hybrid 추가 파라미터
                ('detect.lane.frame_skip',            3),
                ('detect.hough.rho',                1.0),
                ('detect.hough.theta',        np.pi / 180.0),
                ('detect.hough.threshold',          30),
                ('detect.hough.min_line_length',    30),
                ('detect.hough.max_line_gap',       10),
                ('detect.roi.y_offset_ratio',      0.55),        #0.6
                ('detect.roi.gap_ratio',           0.2),        #0.2
            ]
        )

                # === Kalman filter 파라미터 ===
        self.declare_parameter('kalman.process_noise',     1e-2)    #기본값 1e-5
        self.declare_parameter('kalman.measurement_noise', 1e-2)    #기본값 1e-1
        self.declare_parameter('kalman.error_cov_post',    1.0)     #기본값 1.0

         # … 기존 Subscriber/Publisher 설정 …
        # === lane_point 퍼블리셔 (칼만 필터 결과) ===
        self.pub_lane_point = self.create_publisher(
            PointStamped, '/detect/lane_point', 1)

        # === debug image 퍼블리셔 ===
        self.pub_debug_image = self.create_publisher(
            Image, '/detect/debug_image', 1)

        # === 2) Subscriber / Publisher 설정 (raw only) :contentReference[oaicite:1]{index=1} ===
        self.bridge = CvBridge()
        self.sub_image = self.create_subscription(
            Image, '/detect/image_input', self.cbFindLane, 1)
        self.pub_image_lane = self.create_publisher(
            Image, '/detect/image_output', 1)
        self.pub_lane  = self.create_publisher(
            Float64, '/detect/lane', 1)
        self.pub_yrel  = self.create_publisher(
            UInt8,   '/detect/yellow_line_reliability', 1)
        self.pub_wrel  = self.create_publisher(
            UInt8,   '/detect/white_line_reliability', 1)
        self.pub_state = self.create_publisher(
            UInt8,   '/detect/lane_state', 1)
        # 마스크 영상도 서브토픽으로 퍼블리시
        self.pub_image_white  = self.create_publisher(
            Image, '/detect/image_output_sub1', 1)
        self.pub_image_yellow = self.create_publisher(
            Image, '/detect/image_output_sub2', 1)

        # === 3) 내부 상태 버퍼 ===
        self.hist_l = deque(maxlen=5)
        self.hist_r = deque(maxlen=5)
        self.prev_l = 0
        self.prev_r = 640
        self.frame_count = 0

        # === 칼만 필터 초기화 (state=[x,y,vx,vy], meas=[x,y]) ===
        pn = self.get_parameter('kalman.process_noise').value
        mn = self.get_parameter('kalman.measurement_noise').value
        pc = self.get_parameter('kalman.error_cov_post').value
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1,0,1,0],
             [0,1,0,1],
             [0,0,1,0],
             [0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array(
            [[1,0,0,0],
             [0,1,0,0]], np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * pn
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * mn
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32) * pc
        self.kf.statePost           = np.zeros((4,1), dtype=np.float32)

        self.get_logger().info('Hybrid DetectLane node initialized')

    def cbFindLane(self, msg):
        # 프레임 스킵
        skip = self.get_parameter('detect.lane.frame_skip').value
        self.frame_count = (self.frame_count + 1) % skip
        if self.frame_count != 0:
            return

        # 이미지 → BGR
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w  = frame.shape[:2]

        # 1) 색상 필터링 → 마스크 + 퍼블리시 :contentReference[oaicite:2]{index=2}
        white_frac, mask_w = self.maskWhiteLane(frame)
        yellow_frac, mask_y = self.maskYellowLane(frame)
        self.pub_image_white.publish(self.bridge.cv2_to_imgmsg(mask_w, 'mono8'))
        self.pub_image_yellow.publish(self.bridge.cv2_to_imgmsg(mask_y,'mono8'))
        color_mask = cv2.bitwise_or(mask_w, mask_y)

        # 2) 에지 검출 + AND
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur      = cv2.GaussianBlur(gray, (5,5), 0)
        edges     = cv2.Canny(blur, 60, 70)
        edges_mask = cv2.bitwise_and(edges, color_mask)

        # 3) ROI 추출
        y0  = int(self.get_parameter('detect.roi.y_offset_ratio').value * h)
        gap = int(self.get_parameter('detect.roi.gap_ratio').value * h)
        roi = edges_mask[y0:y0+gap, :]

        # 4) HoughLinesP → l_h, r_h :contentReference[oaicite:3]{index=3}
        l_h, r_h = self.hough_lines(roi, w)

        # 5) PolyFit → l_c, r_c, valid_c
        l_c, r_c, valid_c = self.poly_fit(color_mask)

        # 6) 융합: valid_c 우선
        lpos, rpos = (l_c, r_c) if valid_c else (l_h, r_h)

        if valid_c:
            # 곡선 감지 성공 시에도 내측 차선 중심에 더 높은 가중치
            meas_x = np.float32(0.7 * l_c + 0.3 * r_c)
        else:
            meas_x = np.float32(0.7 * l_h + 0.3 * r_h)

        # === 1) 칼만 필터 Predict & Correct ===
        # raw measurement: 차선 중앙 픽셀
        meas_y = np.float32(frame.shape[0] - 1)  # 바닥선 기준
        pred = self.kf.predict()
        est  = self.kf.correct(np.array([[meas_x], [meas_y]], dtype=np.float32))
        k_x, k_y = float(est[0,0]), float(est[1,0])

        # === 2) lane_point 퍼블리시 ===
        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = 'camera_frame'
        pt.point.x = k_x
        pt.point.y = k_y
        pt.point.z = 0.0
        self.pub_lane_point.publish(pt)

        # === 3) 시각화에 칼만 결과 추가 ===
        # (frame 위에 circle로 표시; draw_results 대체 또는 병합)
        cv2.circle(frame, (int(k_x), int(k_y)), 6, (255, 0, 0), -1)

        # 7) 시각화
        self.make_lane(frame, white_frac, yellow_frac, l_h, r_h, lpos, rpos)

        # 8) 퍼블리시: lane center, reliabilities, state
        self.pub_lane.publish(Float64(data=(lpos+rpos)/2.0))
        self.pub_wrel.publish(UInt8(data=white_frac))
        self.pub_yrel.publish(UInt8(data=yellow_frac))
        # state 정의: 0:none,1:left,2:both,3:right
        state = 2 if valid_c else (1 if l_h < w/2 else 3 if r_h > w/2 else 0)
        self.pub_state.publish(UInt8(data=state))

        # 9) 출력 이미지 퍼블리시
        out = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        self.pub_image_lane.publish(out)

        # === 4) debug_image 퍼블리시 ===
        dbg_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        dbg_msg.header = pt.header
        self.pub_debug_image.publish(dbg_msg)

    def maskWhiteLane(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hl = self.get_parameter('detect.lane.white.hue_l').value
        hh = self.get_parameter('detect.lane.white.hue_h').value
        sl = self.get_parameter('detect.lane.white.saturation_l').value
        sh = self.get_parameter('detect.lane.white.saturation_h').value
        ll = self.get_parameter('detect.lane.white.lightness_l').value
        lh = self.get_parameter('detect.lane.white.lightness_h').value
        lower = np.array([hl, sl, ll]); upper = np.array([hh, sh, lh])
        mask  = cv2.inRange(hsv, lower, upper)
        frac  = int(100 * cv2.countNonZero(mask) / mask.size)
        return frac, mask

    def maskYellowLane(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hl = self.get_parameter('detect.lane.yellow.hue_l').value
        hh = self.get_parameter('detect.lane.yellow.hue_h').value
        sl = self.get_parameter('detect.lane.yellow.saturation_l').value
        sh = self.get_parameter('detect.lane.yellow.saturation_h').value
        ll = self.get_parameter('detect.lane.yellow.lightness_l').value
        lh = self.get_parameter('detect.lane.yellow.lightness_h').value
        lower = np.array([hl, sl, ll]); upper = np.array([hh, sh, lh])
        mask  = cv2.inRange(hsv, lower, upper)
        frac  = int(100 * cv2.countNonZero(mask) / mask.size)
        return frac, mask

    def hough_lines(self, roi, width):
        rho    = self.get_parameter('detect.hough.rho').value
        theta  = self.get_parameter('detect.hough.theta').value
        thresh = self.get_parameter('detect.hough.threshold').value
        minl   = self.get_parameter('detect.hough.min_line_length').value
        maxg   = self.get_parameter('detect.hough.max_line_gap').value
        lines  = cv2.HoughLinesP(roi, rho, theta, thresh,
                                 minLineLength=minl, maxLineGap=maxg)
        if lines is None:
            return self.prev_l, self.prev_r
        pts = lines.reshape(-1,4)
        lefts, rights = [], []
        for x1,y1,x2,y2 in pts:
            slope = (y2-y1)/(x2-x1+1e-6)
            midx  = (x1+x2)//2
            if slope < -0.3: lefts.append(midx)
            elif slope >  0.3: rights.append(midx)
        raw_l = int(np.mean(lefts))  if lefts  else self.prev_l
        raw_r = int(np.mean(rights)) if rights else self.prev_r
        self.hist_l.append(raw_l); self.hist_r.append(raw_r)
        lpos = int(np.mean(self.hist_l)); rpos = int(np.mean(self.hist_r))
        self.prev_l, self.prev_r = lpos, rpos
        return lpos, rpos

    def poly_fit(self, mask):
        h, w = mask.shape
        hist = np.sum(mask[h//2:,:], axis=0)
        mid = w//2
        base_l = np.argmax(hist[:mid])
        base_r = np.argmax(hist[mid:]) + mid
        nw, wh = 9, h//9; margin, minpix = 50, 50
        ys, xs = mask.nonzero()
        cur_l, cur_r = base_l, base_r
        inds_l, inds_r = [], []
        for i in range(nw):
            y_low  = h - (i+1)*wh; y_high = h - i*wh
            xl, xh = cur_l-margin, cur_l+margin
            rl, rh = cur_r-margin, cur_r+margin
            good_l = ((ys>=y_low)&(ys<y_high)&(xs>=xl)&(xs<xh)).nonzero()[0]
            good_r = ((ys>=y_low)&(ys<y_high)&(xs>=rl)&(xs<rh)).nonzero()[0]
            if len(good_l)>minpix: cur_l = int(xs[good_l].mean())
            if len(good_r)>minpix: cur_r = int(xs[good_r].mean())
            inds_l.append(good_l); inds_r.append(good_r)
        inds_l = np.concatenate(inds_l); inds_r = np.concatenate(inds_r)
        if len(inds_l)<minpix or len(inds_r)<minpix:
            return None, None, False
        yl, xl = ys[inds_l], xs[inds_l]
        yr, xr = ys[inds_r], xs[inds_r]
        fit_l = np.polyfit(yl, xl, 2); fit_r = np.polyfit(yr, xr, 2)
        l_final = int(fit_l[0]*h*h + fit_l[1]*h + fit_l[2])
        r_final = int(fit_r[0]*h*h + fit_r[1]*h + fit_r[2])
        return l_final, r_final, True

    def make_lane(self, frame, white_frac, yellow_frac, l_h, r_h, lpos, rpos):
        h, w = frame.shape[:2]
        baseline = int(h * 0.95)
        # ROI box
        y0  = int(self.get_parameter('detect.roi.y_offset_ratio').value * h)
        gap = int(self.get_parameter('detect.roi.gap_ratio').value * h)
        cv2.rectangle(frame, (0,y0), (w,y0+gap), (255,0,0), 2)
        # Hough raw
        cv2.circle(frame, (l_h, baseline), 7, (0,255,255), -1)
        cv2.circle(frame, (r_h, baseline), 7, (255,0,255), -1)
        # Final
        cv2.circle(frame, (lpos, baseline), 7, (0,255,0), -1)
        cv2.circle(frame, (rpos, baseline), 7, (0,0,255), -1)
        # Center line
        cv2.line(frame, (lpos,h-1), (rpos,h-1), (0,255,255), 2)

def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
