import cv2
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategorical.h5')

image=cv2.imread('C:\\Users\merci\Desktop\Tumor Detection\\pred\\pred0.jpg') 

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)


result=model.predict(input_img) 
classes_x=np.argmax(result,axis=1)

print(classes_x)

