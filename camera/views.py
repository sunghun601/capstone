from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2

RTSP_STREAM_URL = "rtsp://172.30.134.147:8080/h264_ulaw.sdp"  # 핸드폰 RTSP URL로 변경

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(RTSP_STREAM_URL)
        if not self.video.isOpened():
            raise RuntimeError('Could not start video capture.')
        # 버퍼 비활성화
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            raise RuntimeError('Could not read frame from video stream.')
        # 프레임 크기 조정
        image = cv2.resize(image, (640, 480))
        ret, jpeg = cv2.imencode('.jpg', image)
        if not ret:
            raise RuntimeError('Could not encode image to JPEG.')
        return jpeg.tobytes()

def index(request):
    return render(request, 'camera/index.html')

def gen(camera):
    while True:
        try:
            frame = camera.get_frame()
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
