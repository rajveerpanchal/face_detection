from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
from django.conf import settings


# def load_face_embeddings(image_dir):
#     embeddings = {}

#     for file in os.listdir(image_dir):
#         img_path = image_dir + file
#         image = cv2.imread(img_path)
#         faces = FaceDetection.detect_faces(image, display_image)
#         x, y, w, h = faces[0]
#         image = image[y:y + h, x:x + w]
#         embeddings[file.split(".")[0]] = FaceDetection.v.img_to_encoding(cv2.resize(image, (160, 160)), FaceDetection.image_size)

#     return embeddings

def embeddings():

	import pdb;pdb.set_trace()
	
	print("loading face detector...")
	protoPath = os.path.sep.join([settings.BASE_DIR, "face_detection_model\\deploy.prototxt"])
	modelPath = os.path.sep.join([settings.BASE_DIR,"face_detection_model\\res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	
	print("loading face recognizer...")
	embedder = cv2.dnn.readNetFromTorch(os.path.join(settings.BASE_DIR,'face_detection_model\\openface_nn4.small2.v1.t7'))

	
	print("quantifying faces...")
	dataset = os.path.sep.join([settings.BASE_DIR, "dataset"])
	imagePaths = list(paths.list_images(dataset))
	knownEmbeddings = []
	knownNames = []

	
	total = 0

	
	for (i, imagePath) in enumerate(imagePaths):
		
		print("processing image {}/{}".format(i + 1,
			len(imagePaths)))
		name = imagePath.split(os.path.sep)[-2]
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]

		
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()


		
		if len(detections) > 0:
			
			i = np.argmax(detections[0, 0, :, 2])
			confidence = detections[0, 0, i, 2]

		
			if confidence > 0.5:
				
				# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				
				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				
				knownNames.append(name)
				knownEmbeddings.append(vec.flatten())
				total += 1

	
	print("serializing {} encodings...".format(total))
	data = {"embeddings": knownEmbeddings, "names": knownNames}
	embeddings = os.path.sep.join([settings.BASE_DIR, "output\\embeddings.pickle"])
	f = open(embeddings, "wb")
	f.write(pickle.dumps(data))
	f.close()



	# def fetch_detections(image, embeddings):
    # faces = FaceDetection.detect_faces(image)
   
    # detections = []
    # for face in faces:
    #     x, y, w, h = face
    #     im_face = image[y:y + h, x:x + w]
    #     img = cv2.resize(im_face, (200, 200))
    #     user_embed = FaceDetection.v.img_to_encoding(cv2.resize(img, (160, 160)), FaceDetection.image_size)
        
    #     detected = {}
    #     for _user in embeddings:
    #         flag, thresh = FaceDetection.is_same(embeddings[_user], user_embed)
    #         if flag:
    #             detected[_user] = thresh
        
    #     detected = {k: v for k, v in sorted(detected.items(), key=lambda item: item[1])}
    #     detected = list(detected.keys())
    #     if len(detected) > 0:
    #         detections.append(detected[0])

    # return detections