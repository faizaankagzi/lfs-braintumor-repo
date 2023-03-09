import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from keras.models import load_model

IMAGE_SIZE = 128

#testing using healthy dara
img = cv2.imread('C:/Users/matts/OneDrive/Desktop/6_Brain_Tumor_ML_Model/data/pred/pred12.jpg')

# testing patient that has tumor
#img = cv2.imread('C:/Users/matts/OneDrive/Desktop/6_Brain_Tumor_ML_Model/data/yes/y26.jpg')

img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
img1=np.array(img)
img1=img1.reshape(1,128,128,3)
img1.shape

plt.imshow(img)
model=load_model('braintumor_model.h5')
predictions=model.predict(img1)
predict=np.argmax(predictions[0])

#final prediction results printed
if predict==1:
    print("Patient has brain tumor ")
else:
    print("Patient is healthy")