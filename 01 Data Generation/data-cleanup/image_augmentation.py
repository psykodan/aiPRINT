import cv2
import numpy as np
import os
from pathlib import Path
import random
import imutils




font = cv2.FONT_HERSHEY_SIMPLEX

dirIn = f"synth_line/test/"
dirOut = f"augmented_synth_line/test"
os.makedirs(f"{dirOut}/good") 
os.makedirs(f"{dirOut}/over") 
os.makedirs(f"{dirOut}/under") 

labels = ["good","over","under"]

for l in labels:
    img_num = 0
    dirFiles = Path(dirIn).glob(f'{l}/*.jpg')
    for file in dirFiles:

        for i in range(2):
            img = cv2.imread(str(file),cv2.IMREAD_UNCHANGED)

            # Colour Shift
            bgr = img[:,:,0:3]
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(hsv)
            diff_color = random.randint(-90,90)
            hnew = np.mod(h + diff_color, 180).astype(np.uint8)
            hsv_new = cv2.merge([hnew,s,v])
            img = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

            #Flip Image
            if random.randint(0,1) == 1:
                img = cv2.flip(img,0)
            if random.randint(0,1) == 1:
                img = cv2.flip(img,1)


            #Rotate Image
            if random.randint(0,1) == 1:
                img = imutils.rotate(img,random.randint(0,360))

            #Crop Image
            marginx = 100
            marginy = 100

            crop_img = img[marginy:-marginy, marginx:-marginx]
            crop_img = cv2.resize(crop_img, (224, 224), 
               interpolation = cv2.INTER_LINEAR)
            #save output
            cv2.imwrite(f'{dirOut}/{l}/{img_num}.jpg', crop_img)
            img_num += 1

        #Original image
        img = cv2.imread(str(file),cv2.IMREAD_UNCHANGED)
        marginx = 100
        marginy = 100
        crop_img = img[marginy:-marginy, marginx:-marginx]
        crop_img = cv2.resize(crop_img, (224, 224), 
               interpolation = cv2.INTER_LINEAR)
        #save output
        cv2.imwrite(f'{dirOut}/{l}/{img_num}.jpg', crop_img)
        


