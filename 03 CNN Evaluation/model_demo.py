import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# importing libraries 
import cv2 
import numpy as np 
import tensorflow as tf


model = tf.keras.models.load_model('results-transfer/Xception_line.h5')
class_names=["good","over","under"]
class_confidence=["","","",""]
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (100, 100)
# fontScale
fontScale = 0.75
# Blue color in BGR
color = (255, 0, 0)

colours = [(0,255,0),#perfect
            (150,150,255),#over
           (255,150,150)#under
]
# Line thickness of 2 px
thickness = 2
angle = 0

x, y, width, height = 800, 320, 300, 300

# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('vid2_line.mp4') 
  
# Check if camera opened successfully 
if (cap.isOpened()== False): 
    print("Error opening video file") 


frame_num = 0
# Read until video is completed 
while(cap.isOpened()): 
      
# Capture frame-by-frame 
    ret, frame = cap.read() 
    if ret == True: 
    # Display the resulting frame 
        #cv2.imshow('Frame', frame) 
        cropped_frame = frame[y-height:y+height, x-width:x+width]
        resized_frame = cv2.resize(cropped_frame,(224,224))
        if frame_num%3 == 0:
            input_data = np.expand_dims(resized_frame, axis=0)
            predictions = model.predict(input_data, verbose = 0)
            score = tf.nn.softmax(predictions[0])
            str_label = class_names[np.argmax(score)]

        for idx, s in enumerate(score):
            bar = ""
            for i in range(int(float(s)*100)):
                bar+="|"
            class_confidence[idx]=bar


        cv2.putText(frame, "good", (100,100), font, 
                fontScale, colours[0], thickness, cv2.LINE_AA)
        cv2.putText(frame, class_confidence[0], (190,100), font, 
                fontScale, colours[0], thickness, cv2.LINE_AA)
        
        cv2.putText(frame, "over", (100,130), font, 
                fontScale, colours[1], thickness, cv2.LINE_AA)
        cv2.putText(frame, class_confidence[1], (190,130), font, 
                fontScale, colours[1], thickness, cv2.LINE_AA)
        
        cv2.putText(frame, "under", (100,160), font, 
                fontScale, colours[2], thickness, cv2.LINE_AA)
        cv2.putText(frame, class_confidence[2], (190,160), font, 
                fontScale, colours[2], thickness, cv2.LINE_AA)

        

        cv2.imshow('Frame', frame) 
        frame_num += 1
    # Press Q on keyboard to exit 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
  
# Break the loop 
    else: 
        break
  
# When everything done, release 
# the video capture object 
cap.release() 
  
# Closes all the frames 
cv2.destroyAllWindows() 
