# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 19:02:44 2017

@author: asus
"""

import os
import numpy as np
from PIL import Image 
import cv2


path = "dataset"
    
images = []
labels = []

names = []

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

i = 0

for labelname in os.listdir(path):
    
    names.append(labelname)
    
    for imagename in os.listdir(path + "/" +  labelname + "/train"):
    
        image_path = path + "/" +  labelname + "/train" + "/" + imagename
        
        #image_pil = Image.open(image_path).convert('L')
        #image = np.array(image_pil, 'uint8')
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale(image)
        
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(i)
    i += 1    

recognizer = cv2.createLBPHFaceRecognizer()

recognizer.train(images, np.array(labels))

i = 0

total = 0
correct = 0

for labelname in os.listdir(path):
    
    for imagename in os.listdir(path + "/" +  labelname + "/test"):
        
        image_path = path + "/" +  labelname + "/test" + "/" + imagename
        
        #predict_image_pil = Image.open(image_path).convert('L')
        #predict_image = np.array(predict_image_pil, 'uint8')
        
        predict_image = cv2.imread(image_path)
        predict_image = cv2.cvtColor(predict_image, cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale(predict_image)
        
        for (x, y, w, h) in faces:
            total += 1
            label_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
            label_actual = i
            if label_actual == label_predicted:
                print "{} is Correctly Recognized with confidence {}".format(names[label_actual], conf)
                correct += 1
            else:
                print "{} is Incorrectly Recognized as {}".format(names[label_actual], names[label_predicted])
    i += 1
        
print "Accuracy is {}".format(correct * 100.0 / total)


