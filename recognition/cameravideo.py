from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2,os,urllib.request,pickle
import numpy as np
from django.conf import settings
from recognition import extract_embeddings
from recognition import train_model


protoPath = os.path.sep.join([settings.BASE_DIR, "face_detection_model\\deploy.prototxt"])
modelPath = os.path.sep.join([settings.BASE_DIR,"face_detection_model\\res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch(os.path.join(settings.BASE_DIR,'face_detection_model/openface_nn4.small2.v1.t7'))
recognizer = os.path.sep.join([settings.BASE_DIR, "output\\recognizer.pickle"])
recognizer = pickle.loads(open(recognizer, "rb").read())
le = os.path.sep.join([settings.BASE_DIR, "output\\le.pickle"])
le = pickle.loads(open(le, "rb").read())
dataset = os.path.sep.join([settings.BASE_DIR, "dataset"])
user_list = [ f.name for f in os.scandir(dataset) if f.is_dir() ]

class FaceDetect(object):

	def __init__(self):
		extract_embeddings.embeddings()
		train_model.model_train()
		# initialize the video stream.
		self.vs = VideoStream(src=0).start()
		
		self.fps = FPS().start()

	def __del__(self):
		cv2.destroyAllWindows()

	def get_frame(self):
		
		frame = self.vs.read()
		frame = cv2.flip(frame,1)

		
		frame = imutils.resize(frame, width=600)
		(h, w) = frame.shape[:2]

		
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		
		# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()


		
		for i in range(0, detections.shape[2]):
			
			confidence = detections[0, 0, i, 2]

			
			if confidence > 0.5:
				
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				
				if fW < 20 or fH < 20:
					continue

				
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]


				
				text = "{}: {:.2f}%".format(name, proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

		
		self.fps.update()
		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tobytes()
		