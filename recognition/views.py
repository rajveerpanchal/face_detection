from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from recognition.cameravideo import FaceDetect
# Create your views here.
def index(request):
	return render(request, 'recognition/home.html')
# def detect_faces(image):
#     height, width, channels = image.shape

#     blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
#     FaceDetection.net.setInput(blob)
#     detections = FaceDetection.net.forward()

#     faces = []

#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.5:
#             x1 = int(detections[0, 0, i, 3] * width)
#             y1 = int(detections[0, 0, i, 4] * height)
#             x2 = int(detections[0, 0, i, 5] * width)
#             y2 = int(detections[0, 0, i, 6] * height)
#             faces.append([x1, y1, x2 - x1, y2 - y1])

#     return faces

def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
		
def facecam_feed(request):
	return StreamingHttpResponse(gen(FaceDetect()),
					content_type='multipart/x-mixed-replace; boundary=frame')
					





































































					