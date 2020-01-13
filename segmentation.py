# import the necessary packages
import numpy as np
import cv2
#import random
import os

# load image
#image = cv2.imread('road.jpg') 
image = cv2.imread('spz-more.jpg') 

# defining prototext and caffemodel paths
caffeModel = "cityscapes/dilation.caffemodel"
prototextPath = "cityscapes/dilation.prototxt"
height = 1396 
width = 1396 
mean = (104.0, 177.0, 123.0)
threshold = 0.5

# Load Model
#os.environ['OPENCV_OCL4DNN_CONFIG_PATH'] = './opencl'
net = cv2.dnn.readNetFromCaffe(prototextPath,caffeModel)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

# convert to RGB
rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# blob preparation
blob = cv2.dnn.blobFromImage(image,1.0,(width,height),mean)

# passing blob through the network to detect and pridiction
net.setInput(blob)
clock0 = cv2.getTickCount() / cv2.getTickFrequency()
out = net.forward()
clock1 = cv2.getTickCount() / cv2.getTickFrequency()

# process result
print(out.shape,clock1-clock0) #(1, 19, 1024, 1024)
res = np.argmax(out,axis=1)

# display result
#colors = [(random.uniform(0,255),random.uniform(0,255),random.uniform(0,255)) for i in range(19)]
colors = [(64,180,255), (128,0,0), (0,128,0), (128,128,0), (0,0,128), (128,0,128), (0,128,128), (128,128,128), (64,0,0), (192,0,0), (64,128,0), (192,128,0), (64,0,128), (192,0,128), (64,128,128), (192,128,128), (0,64,0), (128,64,0), (0,192,0)]
colors = np.array(colors,np.uint8)
disp = colors[res[0]]
cv2.imwrite('disp.png',disp)

gray = cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
disp2 = np.zeros((height,width,3))
tl=((disp2.shape[0]-1024)//2,(disp2.shape[1]-1024)//2)
disp2[tl[0]:tl[0]+1024,tl[1]:tl[1]+1024] = disp
disp2 = cv2.resize(disp2,(gray.shape[1],gray.shape[0]))
disp2 = cv2.multiply(disp2.astype(np.float32),gray.astype(np.float32)/255.0).astype(np.uint8)
cv2.imwrite('disp2.png',disp2)

