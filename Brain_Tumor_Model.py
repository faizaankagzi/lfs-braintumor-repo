import cv2
import os
import sklearn
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.compat.v1.Session(config=config)
print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))

IMAGE_SIZE = 128
BATCH_SIZE = 16
VERBOSE = 1

class_names = ["no", "yes"]
base_path = "C:/Users/matts/OneDrive/Desktop/6_Brain_Tumor_ML_Model/data"

x_train=[]
y_train=[]

for i in class_names:
    folderPath = os.path.join(base_path,i)
    for j in tqdm(os.listdir(folderPath), ncols=70):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
        x_train.append(img)
        y_train.append(i)
print('Training dataset Loading complete.')

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,horizontal_flip=True, fill_mode="nearest")

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train, y_train = sklearn.utils.shuffle(x_train, y_train, random_state=0)

print('Testing dataset Loading complete.')

#setting up the training and testing split fo rthe dataset uploaded
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=47, test_size=0.20)

y_train_new = [class_names.index(i) for i in y_train]
y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)
y_test_new = [class_names.index(i) for i in y_test]
y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)

#creation of the model
inputShape = (IMAGE_SIZE, IMAGE_SIZE, 3)
braintumor_model = Xception(input_shape=inputShape, include_top=False)
model = braintumor_model.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(0.5)(model)
model = tf.keras.layers.Dense(2, activation='softmax')(model)
model = tf.keras.models.Model(inputs=braintumor_model.input, outputs=model)

# saving the model as .h5 file
tensorboard = TensorBoard(log_dir = 'logs')
checkpoint = ModelCheckpoint("braintumor_model.h5",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,mode='auto',verbose=VERBOSE)

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
history=model.fit(aug.flow(x_train, y_train,batch_size=BATCH_SIZE),
          
validation_data=(x_test, y_test), steps_per_epoch=len(x_train) // BATCH_SIZE,
epochs=3, callbacks=[reduce_lr, checkpoint, tensorboard])
result = model.evaluate(x_test, y_test)
print(result)