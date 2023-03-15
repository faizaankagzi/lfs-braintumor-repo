import tensorflow as tf
import numpy as np
import io

MODEL = tf.keras.models.load_model("braintumor_model.h5")
CLASSES = ["Congratulations! The patient is HEALTHY", "TUMOR DETECTED, Seek medical help!!"]

def load_img(img):
	# Reading The Image
	img = img.file.read()
	# Converting to Bytes Stream
	img = io.BytesIO(img)
	# Loading Image
	img = tf.keras.preprocessing.image.load_img(img, target_size=MODEL.input_shape[1:])
	# Converting Image to Array Of Correct Dimensions
	img = np.expand_dims(img, 0)
	# Returning Loaded Image
	return img

def preprocess(img):
	# Preprocessing the image
	#img = tf.keras.applications.vgg16.preprocess_input(img)
	#img = img / 255.
	#Returning the processed image
	return img

def predict(X):
	pred = MODEL.predict(X)
	i = np.argmax(pred[0])
	return {
		"prediction": CLASSES[i],
		"probability": round(pred[0, i].tolist(), 3)
	}

def pipeline(img):

	loaded_img = load_img(img)
	#processed_img = preprocess(loaded_img)
	response = predict(loaded_img)

	return response
