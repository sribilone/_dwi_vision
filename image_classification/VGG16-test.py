from keras.preprocessing import image
import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt
#img = image.load_img("Incident/validation/Clear/541_visual_20240303_000743.jpg",target_size=(224,224))
#img = image.load_img("Incident/validation/Dark/541_visual_20240305_015203.jpg",target_size=(224,224))
img = image.load_img("Incident/validation/Smoke/541_visual_20240303_000053.jpg",target_size=(224,224))
from keras.models import load_model
saved_model = load_model("checkpoint/vgg16_Classification_e1000_s137_224X224.h5")
#img = image.load_img("Incident/validation/Thermal/541_thermal_20240306_000013.jpg",target_size=(224,224))
List_image_Name = glob.glob("D:/Data_Center/UnburnClassificationIncident/ForTestingClassification/Dark/*.jpg")
f = open('vgg16_Classification_e1000_s137_224X224/vgg16_Classification_e1000_s137_224X224_Dark.txt', 'w')
icount = 1
iNo_Light = 0
iNormal   = 0
iSmoke    = 0
iThermal  = 0
for image_Name in List_image_Name:

    if os.path.isfile(image_Name):
        img = image.load_img(image_Name, target_size=(224, 224))
        img = np.asarray(img)
        #plt.imshow(img)
        img = np.expand_dims(img, axis=0)
        line = ''
        output = saved_model.predict(img)
        if icount > 0 and icount < 10 :
            print(image_Name)
            print(np.argmax(output, axis=1), output)
            icount = icount + 1
        #print(np.argmax(output, axis=1),output)
        if np.argmax(output, axis=1) == 0:
            #print(image_Name,"Clear")
            line = image_Name + ",No_Light"
            iNo_Light = iNo_Light + 1
            f.write(line)
        elif np.argmax(output, axis=1) == 1:
            #print(image_Name,'Dark')
            line = image_Name + ",Normal"
            iNormal = iNormal + 1
            f.write(line)
        elif np.argmax(output, axis=1) == 2:
            #print(image_Name,'Smoke')
            line = image_Name + ",Smoke"
            iSmoke = iSmoke + 1
            f.write(line)
        elif np.argmax(output, axis=1) == 3:
            #print(image_Name,'Thermal')
            line = image_Name + ",Thermal"
            iThermal = iThermal + 1
            f.write(line)

        f.write('/n')

print('vgg16_Classification_e1000_s137_224X224',iNo_Light,iNormal,iSmoke,iThermal)

List_image_Name = glob.glob("D:/Data_Center/UnburnClassificationIncident/ForTestingClassification/Normal/*.jpg")
f = open('vgg16_Classification_e1000_s137_224X224/vgg16_Classification_e1000_s137_224X224_Normal.txt', 'w')
icount = 1
iNo_Light = 0
iNormal   = 0
iSmoke    = 0
iThermal  = 0
for image_Name in List_image_Name:

    if os.path.isfile(image_Name):
        img = image.load_img(image_Name, target_size=(224, 224))
        img = np.asarray(img)
        #plt.imshow(img)
        img = np.expand_dims(img, axis=0)
        line = ''
        output = saved_model.predict(img)
        if icount > 0 and icount < 10 :
            print(image_Name)
            print(np.argmax(output, axis=1), output)
            icount = icount + 1
        #print(np.argmax(output, axis=1),output)
        if np.argmax(output, axis=1) == 0:
            #print(image_Name,"Clear")
            line = image_Name + ",No_Light"
            iNo_Light = iNo_Light + 1
            f.write(line)
        elif np.argmax(output, axis=1) == 1:
            #print(image_Name,'Dark')
            line = image_Name + ",Normal"
            iNormal = iNormal + 1
            f.write(line)
        elif np.argmax(output, axis=1) == 2:
            #print(image_Name,'Smoke')
            line = image_Name + ",Smoke"
            iSmoke = iSmoke + 1
            f.write(line)
        elif np.argmax(output, axis=1) == 3:
            #print(image_Name,'Thermal')
            line = image_Name + ",Thermal"
            iThermal = iThermal + 1
            f.write(line)

        f.write('/n')

print(iNo_Light,iNormal,iSmoke,iThermal)


List_image_Name = glob.glob("D:/Data_Center/UnburnClassificationIncident/ForTestingClassification/Smoke/*.jpg")
f = open('vgg16_Classification_e1000_s137_224X224/vgg16_Classification_e1000_s137_224X224_Smoke.txt', 'w')
icount = 1
iNo_Light = 0
iNormal   = 0
iSmoke    = 0
iThermal  = 0
for image_Name in List_image_Name:

    if os.path.isfile(image_Name):
        img = image.load_img(image_Name, target_size=(224, 224))
        img = np.asarray(img)
        #plt.imshow(img)
        img = np.expand_dims(img, axis=0)
        line = ''
        output = saved_model.predict(img)
        if icount > 0 and icount < 10 :
            print(image_Name)
            print(np.argmax(output, axis=1), output)
            icount = icount + 1
        #print(np.argmax(output, axis=1),output)
        if np.argmax(output, axis=1) == 0:
            #print(image_Name,"Clear")
            line = image_Name + ",No_Light"
            iNo_Light = iNo_Light + 1
            f.write(line)
        elif np.argmax(output, axis=1) == 1:
            #print(image_Name,'Dark')
            line = image_Name + ",Normal"
            iNormal = iNormal + 1
            f.write(line)
        elif np.argmax(output, axis=1) == 2:
            #print(image_Name,'Smoke')
            line = image_Name + ",Smoke"
            iSmoke = iSmoke + 1
            f.write(line)
        elif np.argmax(output, axis=1) == 3:
            #print(image_Name,'Thermal')
            line = image_Name + ",Thermal"
            iThermal = iThermal + 1
            f.write(line)

        f.write('/n')

print(iNo_Light,iNormal,iSmoke,iThermal)


List_image_Name = glob.glob("D:/Data_Center/UnburnClassificationIncident/ForTestingClassification/Thermal/*.jpg")
f = open('vgg16_Classification_e1000_s137_224X224/vgg16_Classification_e1000_s137_224X224_Thermal.txt', 'w')
icount = 1
iNo_Light = 0
iNormal   = 0
iSmoke    = 0
iThermal  = 0
for image_Name in List_image_Name:

    if os.path.isfile(image_Name):
        img = image.load_img(image_Name, target_size=(224, 224))
        img = np.asarray(img)
        #plt.imshow(img)
        img = np.expand_dims(img, axis=0)
        line = ''
        output = saved_model.predict(img)
        if icount > 0 and icount < 10 :
            print(image_Name)
            print(np.argmax(output, axis=1), output)
            icount = icount + 1
        #print(np.argmax(output, axis=1),output)
        if np.argmax(output, axis=1) == 0:
            #print(image_Name,"Clear")
            line = image_Name + ",No_Light"
            iNo_Light = iNo_Light + 1
            f.write(line)
        elif np.argmax(output, axis=1) == 1:
            #print(image_Name,'Dark')
            line = image_Name + ",Normal"
            iNormal = iNormal + 1
            f.write(line)
        elif np.argmax(output, axis=1) == 2:
            #print(image_Name,'Smoke')
            line = image_Name + ",Smoke"
            iSmoke = iSmoke + 1
            f.write(line)
        elif np.argmax(output, axis=1) == 3:
            #print(image_Name,'Thermal')
            line = image_Name + ",Thermal"
            iThermal = iThermal + 1
            f.write(line)

        f.write('/n')

print(iNo_Light,iNormal,iSmoke,iThermal)