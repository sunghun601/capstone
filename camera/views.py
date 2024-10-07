import cv2
import torch
import numpy as np
import time
from django.shortcuts import render
from django.http import StreamingHttpResponse
import sys

YOLO_MODEL_PATH = "C:/git/last.pt"  # YOLOv7 모델 경로
YOLOV7_PATH = "C:/git/msi/yolov7"

# YOLOv7 저장소 경로를 시스템 경로에 추가
sys.path.append(YOLOV7_PATH)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox


class VideoCamera(object):
    def __init__(self):
        # USB로 연결된 첫 번째 카메라 사용 (DirectShow 백엔드 사용)
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.video.isOpened():
            raise RuntimeError('Could not start video capture.')

        # 해상도와 FPS 설정
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 15)  # FPS를 15로 설정

        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 0)

        # YOLOv7 모델 로드 (weights_only 인수 제거)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = attempt_load(YOLO_MODEL_PATH, map_location=self.device)  # weights_only 인수 제거
        self.model.eval()

        self.frame_interval = 3  # 처리할 프레임 간격
        self.frame_count = 0
        self.last_frame = None  # 초기화

        # FPS 측정 변수
        self.prev_time = time.time()
        self.fps = 0

    def __del__(self):
        if self.video.isOpened():
            self.video.release()  # 비디오 캡처 객체 해제
        cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기

    def get_frame(self):
        retry_count = 0
        while retry_count < 5:  # 최대 5회까지 재시도
            success, image = self.video.read()
            if success:
                print(f"Frame read successfully: {retry_count + 1} retry attempt(s)")
                break
            else:
                print(f"Failed to read frame: Retry attempt {retry_count + 1}")
                retry_count += 1
                time.sleep(1)  # 1초 대기 후 재시도
        if not success:
            print("Unable to read frame after 5 attempts. Reinitializing camera...")
            self.__del__()  # 스트림 해제
            self.__init__()  # 스트림 재시작

        # FPS 계산
        curr_time = time.time()
        time_diff = curr_time - self.prev_time
        if time_diff == 0:
            time_diff = 1e-6  # 0으로 나누는 문제 방지
        self.fps = 1.0 / time_diff
        self.prev_time = curr_time

        # 프레임에 FPS 표시
        cv2.putText(image, f'FPS: {self.fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.frame_count += 1
        if self.frame_count % self.frame_interval != 0:
            return self.last_frame

        # YOLOv7을 이용한 객체 탐지
        img = letterbox(image, 640, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # 강제로 float 타입으로 변경
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.model.names[int(cls)]}: {conf:.2f}'  # 클래스 이름과 확률
                    cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(image, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        if not ret:
            print("Failed to encode image to JPEG. Retrying...")
            return self.last_frame
        self.last_frame = jpeg.tobytes()  # 마지막 프레임 업데이트
        return self.last_frame


def index(request):
    return render(request, 'camera/camera.html')


def gen(camera):
    while True:
        try:
            frame = camera.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except RuntimeError as e:
            print(f"Error: {e}")
            break


def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def recycle_chart(request):
    return render(request, 'camera/recycle_chart.html')
