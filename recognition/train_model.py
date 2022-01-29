from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle,os
from django.conf import settings

def model_train():
	
	print(" loading face embeddings...")
	embeddings = os.path.sep.join([settings.BASE_DIR, "output\\embeddings.pickle"])
	data = pickle.loads(open(embeddings, "rb").read())
	print("encoding labels...")
	le = LabelEncoder()
	labels = le.fit_transform(data["names"])
	print("training model...")
	recognizer = SVC(C=1.0, kernel="linear", probability=True)
	recognizer.fit(data["embeddings"], labels)
	recognizers = os.path.sep.join([settings.BASE_DIR, "output\\recognizer.pickle"])
	f = open(recognizers, "wb")
	f.write(pickle.dumps(recognizer))
	f.close()
	le_pickle = os.path.sep.join([settings.BASE_DIR, "output\\le.pickle"])
	f = open(le_pickle, "wb")
	f.write(pickle.dumps(le))
	f.close()