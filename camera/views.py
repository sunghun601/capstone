import cv2
import torch
import numpy as np
from django.shortcuts import render
from django.http import StreamingHttpResponse
import sys

RTSP_STREAM_URL = "rtsp://172.30.137.245:8080/h264_ulaw.sdp"  # 핸드폰 RTSP URL로 변경
YOLO_MODEL_PATH = "/Users/kimsunghun/Desktop/samplemodel.pt"  # YOLOv7 모델 경로
YOLOV7_PATH = "/Users/kimsunghun/PycharmProjects/msi/mysite/yolov7"

# YOLOv7 저장소 경로를 시스템 경로에 추가
sys.path.append(YOLOV7_PATH)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(RTSP_STREAM_URL)
        if not self.video.isOpened():
            raise RuntimeError('Could not start video capture.')
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # YOLOv7 모델 로드
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = attempt_load(YOLO_MODEL_PATH, map_location=self.device)
        self.model.eval()

        self.frame_interval = 3  # 처리할 프레임 간격
        self.frame_count = 0
        self.last_frame = None  # 초기화

    def __del__(self):
        self.video.release()

    def get_frame(self):
        while True:
            success, image = self.video.read()
            if not success:
                print("Failed to read frame from video stream. Retrying...")
                continue
            image = cv2.resize(image, (320, 240))  # 해상도 축소

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
                        cv2.putText(image, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            ret, jpeg = cv2.imencode('.jpg', image)
            if not ret:
                print("Failed to encode image to JPEG. Retrying...")
                continue
            self.last_frame = jpeg.tobytes()  # 마지막 프레임 업데이트
            return self.last_frame

def index(request):
    return render(request, 'camera/index.html')

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
