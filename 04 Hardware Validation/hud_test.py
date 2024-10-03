import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# importing libraries 
import cv2 
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import time

#Video recording
fourcc = cv2.VideoWriter_fourcc(*'h264')
out = cv2.VideoWriter('Examplehgh.mp4', fourcc, 20.0, (1280,720))
capture_num=0

models ={"vgg19_line" : tf.keras.models.load_model('/Users/daniel/Documents/aiPrint-Publishable/aiPrint-Data-Creation/cnn-training/results-real/Xception_line.h5')}
class_names=["good","over","under"]
class_confidence=["","","",""]
detection_count = [0,0,0]
x, y, cnnwidth, cnnheight = 700, 370, 300, 300

# font
font = cv2.FONT_HERSHEY_DUPLEX
# org
org = (100, 100)
# fontScale
fontScale = 0.75
# Line thickness of 2 px
thickness = 2
angle = 0

colours = {"good":(0,200,0),
            "over":(0,0,200),
           "under":(200,0,0),
           "HUD":(100, 100, 100)
}


length, width = 1280, 720
c_length = int(length/2)
c_width = int(width/2)
top_right = 250
top_left = 150
# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('vid2_line.mp4') 

# Check if camera opened successfully 
if (cap.isOpened()== False): 
    print("Error opening video file") 


label=""
colour = (0,0,0)
frame_num = 0
fps = 0
# Read until video is completed 
start = time.time()
while(cap.isOpened()): 
      
# Capture frame-by-frame 
    ret, frame = cap.read() 
    rawframe = frame
    if ret == True: 
    # Display the resulting frame 
        #cv2.imshow('Frame', frame)
        frame = cv2.resize(frame,(length,width))
        cropped_frame = frame[y-cnnheight:y+cnnheight, x-cnnwidth:x+cnnwidth]
        resized_frame = cv2.resize(cropped_frame,(140,140))
        resized_frame = cv2.resize(cropped_frame,(224,224))
        
        if frame_num%10 == 0:
            details=""
            votes = []
            scores = []
            input_data = np.expand_dims(resized_frame, axis=0)
            for model in models:
                predictions = models[model].predict(input_data, verbose = 0)
                score = tf.nn.softmax(predictions[0])
                votes.append([class_names[np.argmax(score)], score[np.argmax(score)]])
                scores.append(predictions[0])
                details += f"{model} good {predictions[0][0]:.5f} over {predictions[0][1]:.5f} under {predictions[0][2]:.5f}\n"

        #Worst case classification
        if scores[0][0] < 0.8:
            c = np.argmax([scores[0][1],scores[0][2]])+1
            label = class_names[c]
            detection_count[c] +=1
        else:
            label = "good"
            detection_count[0] +=1
        
        #Centre view finder
        #top left
        frame = cv2.line(frame,(c_length-250,c_width-250), (c_length-250,c_width-200), colours["HUD"],1)
        frame = cv2.line(frame,(c_length-250,c_width-250), (c_length-200,c_width-250), colours["HUD"],1)
        frame = cv2.line(frame,(c_length-305,c_width-305), (c_length-305,c_width-200), colours["HUD"],1)
        frame = cv2.line(frame,(c_length-305,c_width-305), (c_length-200,c_width-305), colours["HUD"],1)
        #top right
        frame = cv2.line(frame,(c_length+250,c_width-250), (c_length+250,c_width-200), colours["HUD"],1)
        frame = cv2.line(frame,(c_length+250,c_width-250), (c_length+200,c_width-250), colours["HUD"],1)
        frame = cv2.line(frame,(c_length+305,c_width-305), (c_length+305,c_width-200), colours["HUD"],1)
        frame = cv2.line(frame,(c_length+305,c_width-305), (c_length+200,c_width-305), colours["HUD"],1)
        #bottom left
        frame = cv2.line(frame,(c_length-250,c_width+250), (c_length-250,c_width+200), colours["HUD"],1)
        frame = cv2.line(frame,(c_length-250,c_width+250), (c_length-200,c_width+250), colours["HUD"],1)
        frame = cv2.line(frame,(c_length-305,c_width+305), (c_length-305,c_width+200), colours["HUD"],1)
        frame = cv2.line(frame,(c_length-305,c_width+305), (c_length-200,c_width+305), colours["HUD"],1)
        #bottom right
        frame = cv2.line(frame,(c_length+250,c_width+250), (c_length+250,c_width+200), colours["HUD"],1)
        frame = cv2.line(frame,(c_length+250,c_width+250), (c_length+200,c_width+250), colours["HUD"],1)
        frame = cv2.line(frame,(c_length+305,c_width+305), (c_length+305,c_width+200), colours["HUD"],1)
        frame = cv2.line(frame,(c_length+305,c_width+305), (c_length+200,c_width+305), colours["HUD"],1)

        #Border
        #left
        frame = cv2.line(frame,(10,10), (10,width-10), colours["HUD"],1)
        frame = cv2.line(frame,(10,10), (20,10), colours["HUD"],1)
        frame = cv2.line(frame,(10,width-10), (20,width-10), colours["HUD"],1)
        #right
        frame = cv2.line(frame,(length-10,10), (length-10,width-10), colours["HUD"],1)
        frame = cv2.line(frame,(length-10,10), (length-20,10), colours["HUD"],1)
        frame = cv2.line(frame,(length-10,width-10), (length-20,width-10), colours["HUD"],1)
        #top
        frame = cv2.ellipse(frame, (c_length,0), (int(length/2),75), 0, 0, 360, colours["HUD"], 1) 
        #bottom
        frame = cv2.ellipse(frame, (c_length,width), (int(length/2),75), 0, 0, 360, colours["HUD"], 1) 

        #Right side detail
        frame = cv2.line(frame,(length-200,top_right-30), (length-200,top_right-100), colours["HUD"],1)
        frame = cv2.line(frame,(length-200,top_right-100),( length-300,top_right-100),  colours["HUD"],1)
        frame = cv2.line(frame,(length-200,top_right+270), (length-200,top_right+340), colours["HUD"],1)
        frame = cv2.line(frame,(length-200,top_right+340),( length-300,top_right+340),  colours["HUD"],1)

        #Right side text
        cv2.putText(frame, f"Time : {datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')}", (length-275,top_right), font, .5, colour, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Frame : {frame_num}", (length-275,top_right+30), font, .5, colour, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS : {fps:.1f}", (length-275,top_right+60), font, .5, colour, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Extrusion Factor : +0%", (length-275,top_right+120), font, .5, colour, 1, cv2.LINE_AA)
        cv2.putText(frame, "Accumulative Error:", (length-275,top_right+150), font, .5, colour, 1, cv2.LINE_AA)
        cv2.putText(frame, f"- Over: {(100 / sum(detection_count)) * (detection_count[1]):.2f}%", (length-275,top_right+180), font, .5, colour, 1, cv2.LINE_AA)
        cv2.putText(frame, f"- Under: {(100 / sum(detection_count)) * (detection_count[2]):.2f}%", (length-275,top_right+210), font, .5, colour, 1, cv2.LINE_AA)
        cv2.putText(frame, f"- Total: {(100 / sum(detection_count)) * (detection_count[1]+detection_count[2]):.2f}%", (length-275,top_right+240), font, .5, colour, 1, cv2.LINE_AA)

        #Confidence bars
        for idx, s in enumerate(predictions[0]):
            bar = ""
            for i in range(int(float(s)*30)):
                bar+="|"
            class_confidence[idx]=bar
        
        cv2.putText(frame, "Classification Confidence", (50,top_left), font, .5, colour, 1, cv2.LINE_AA)
        cv2.putText(frame, f"good:{predictions[0][0]:.3f}", (50,top_left+30), font, .5, colours["good"], 1, cv2.LINE_AA)
        cv2.putText(frame, class_confidence[0], (150,top_left+30), font, .5, colours["good"], 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"over:{predictions[0][1]:.3f}", (50,top_left+60), font, .5, colours["over"], 1, cv2.LINE_AA)
        cv2.putText(frame, class_confidence[1], (150,top_left+60), font, .5, colours["over"], 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"under:{predictions[0][2]:.3f}", (50,top_left+90), font, .5, colours["under"], 1, cv2.LINE_AA)
        cv2.putText(frame, class_confidence[2], (150,top_left+90), font, .5, colours["under"], 1, cv2.LINE_AA)
        
        #Detection bars
        bar_len = 200 / sum(detection_count)
        good_len = int(bar_len * detection_count[0])
        over_len = int(bar_len * detection_count[1])
        under_len = int(bar_len * detection_count[2])
        cv2.putText(frame, "Classification Total Split", (50,top_left+150), font, .5, colour, 1, cv2.LINE_AA)
        frame = cv2.rectangle(frame,(50,top_left+180),(50+good_len,top_left+280), colours["good"],-1)
        frame = cv2.rectangle(frame,(50+good_len,top_left+180),(50+good_len+over_len,top_left+280), colours["over"],-1)
        frame = cv2.rectangle(frame,(50+good_len+over_len,top_left+180),(50+good_len+over_len+under_len,top_left+280), colours["under"],-1)

        #Detection Label
        cv2.putText(frame, "Current Classification", (50,top_left+340), font, .5, colour, 1, cv2.LINE_AA)
        if(label=="good"):
            cv2.putText(frame, label, (80,top_left+395), font, 1, colours["good"], 1, cv2.LINE_AA)
        elif(label=="over"):
            cv2.putText(frame, label, (80,top_left+395), font, 1, colours["over"], 1, cv2.LINE_AA)
        elif(label=="under"):
            cv2.putText(frame, label, (80,top_left+395), font, 1, colours["under"], 1, cv2.LINE_AA)
        frame = cv2.rectangle(frame,(50,top_left+360),(200,top_left+415), colours["HUD"],1)
        frame = cv2.line(frame,(125,top_left+415), (125,top_left+465), colours["HUD"],1)
        frame = cv2.line(frame,(125,top_left+465), (c_length-250-25,top_left+465), colours["HUD"],1)
        frame = cv2.circle(frame,(c_length-250,c_width+250),25, colours["HUD"],1)

        cv2.imshow('Frame', frame) 
        #out.write(frame)
        frame_num += 1
        fps = frame_num / (time.time()-start)
    # Press Q on keyboard to exit 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
        
        if cv2.waitKey(25) & 0xFF == ord('c'): 
            cv2.imwrite("hud.png",frame)
            cv2.imwrite("raw.png",rawframe)
 
  
# Break the loop 
    else: 
        break
  
# When everything done, release 
# the video capture object 
cap.release() 
#out.release()
# Closes all the frames 
cv2.destroyAllWindows() 
