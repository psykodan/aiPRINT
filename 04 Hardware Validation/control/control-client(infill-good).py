import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# importing libraries 
import cv2 
import numpy as np 
import socket
import datetime
import time
import json
from picamera2 import Picamera2
import serial
from threading import Thread

PATH = "/home/pi"

#Camera image capture setup
picam = Picamera2()
config = picam.create_still_configuration(controls={'ExposureTime':5000,'AnalogueGain':18.0},main={"size":(1280,720)})
picam.configure(config)
picam.start()

#Connection to Tensorflow process machine setup
UDP_IP = "10.0.2.79" # set it to destination IP.. RPi in this case
UDP_PORT = 12345
print("UDP target IP:", UDP_IP)
print("UDP target port:", UDP_PORT)
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
sock.settimeout(10.0)


#Onscreen Text setup
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
           "HUD":(200, 0, 200)
}
length, width = 1280, 720
c_length = int(length/2)
c_width = int(width/2)
top_right = 250
top_left = 150

#Video recording
fourcc = cv2.VideoWriter_fourcc(*'h264')
out = cv2.VideoWriter('videos/control_test_c_good.mp4', fourcc, 14.0, (1280,720))

#Region of Interest
box = 220
boxy = 340
boxx=650


extrusion_factor = 1
print_start = False

#Camera feed and transmission of classification
def camera_system():
    global extrusion_factor,print_start
    resp=["","","","","","","","","","",1,1,1]

    #Information display
    class_names=["good","over","under"]
    class_confidence=["","","",""]
    detection_count = [0,0,0]
    label=""
    colour = (0,0,0)
    frame_num = 0
    fps = 0
    
    # Read until video is completed 
    start = time.time()
    while(True): 
        
    # Capture frame-by-frame 
        frame = picam.capture_array()
        #print("Cap")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.resize(frame,(1280,720))
        #cv2.imshow('frame',frame)
    # Display the resulting frame 
        #cv2.imshow('Frame', frame)
        if print_start == True:
            if frame_num%15==0:
                data = frame[boxy-box:boxy+box,boxx-box:boxx+box]
                data = cv2.resize(data,(140,140))
                
                data = data.reshape(-1)

                sock.sendto(data.tobytes(),(UDP_IP,UDP_PORT))

                resp, server = sock.recvfrom(512)
                resp = json.loads(resp.decode())
                if extrusion_factor >= 0.2 and extrusion_factor <=2.5:
                    if resp[0] == "over":
                        extrusion_factor -= 0.15
                    elif resp[0] == "under":
                        extrusion_factor += 0.15
            if extrusion_factor <0.2:
                extrusion_factor = 0.2
            if extrusion_factor >2.5:
                extrusion_factor = 2.5
                    
                
            

            
            
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
            cv2.putText(frame, f"Extrusion Factor : {extrusion_factor*100}%", (length-275,top_right+120), font, .5, colour, 1, cv2.LINE_AA)
            cv2.putText(frame, "Accumulative Error:", (length-275,top_right+150), font, .5, colour, 1, cv2.LINE_AA)
            cv2.putText(frame, f"- Over: {resp[7]}", (length-275,top_right+180), font, .5, colour, 1, cv2.LINE_AA)
            cv2.putText(frame, f"- Under: {resp[8]}", (length-275,top_right+210), font, .5, colour, 1, cv2.LINE_AA)
            cv2.putText(frame, f"- Good: {resp[9]}", (length-275,top_right+240), font, .5, colour, 1, cv2.LINE_AA)
            
            #Confidence bars
            cv2.putText(frame, "Classification Confidence", (50,top_left), font, .5, colour, 1, cv2.LINE_AA)
            cv2.putText(frame, f"good:{resp[1]}", (50,top_left+30), font, .5, colours["good"], 1, cv2.LINE_AA)
            cv2.putText(frame, resp[4], (150,top_left+30), font, .5, colours["good"], 1, cv2.LINE_AA)
            
            cv2.putText(frame, f"over:{resp[2]}", (50,top_left+60), font, .5, colours["over"], 1, cv2.LINE_AA)
            cv2.putText(frame, resp[5], (150,top_left+60), font, .5, colours["over"], 1, cv2.LINE_AA)
            
            cv2.putText(frame, f"under:{resp[3]}", (50,top_left+90), font, .5, colours["under"], 1, cv2.LINE_AA)
            cv2.putText(frame, resp[6], (150,top_left+90), font, .5, colours["under"], 1, cv2.LINE_AA)
            
            #Detection bars
            bar_len = 200 / (resp[10]+resp[11]+resp[12])
            good_len = int(bar_len * resp[10])
            over_len = int(bar_len * resp[11])
            under_len = int(bar_len * resp[12])
            cv2.putText(frame, "Classification Total Split", (50,top_left+150), font, .5, colour, 1, cv2.LINE_AA)
            frame = cv2.rectangle(frame,(50,top_left+180),(50+good_len,top_left+280), colours["good"],-1)
            frame = cv2.rectangle(frame,(50+good_len,top_left+180),(50+good_len+over_len,top_left+280), colours["over"],-1)
            frame = cv2.rectangle(frame,(50+good_len+over_len,top_left+180),(50+good_len+over_len+under_len,top_left+280), colours["under"],-1)
            
            #Detection Label
            cv2.putText(frame, "Current Classification", (50,top_left+340), font, .5, colour, 1, cv2.LINE_AA)
            if(resp[0]=="good"):
                cv2.putText(frame, resp[0], (80,top_left+395), font, 1, colours["good"], 1, cv2.LINE_AA)
            elif(resp[0]=="over"):
                cv2.putText(frame, resp[0], (80,top_left+395), font, 1, colours["over"], 1, cv2.LINE_AA)
            elif(resp[0]=="under"):
                cv2.putText(frame, resp[0], (80,top_left+395), font, 1, colours["under"], 1, cv2.LINE_AA)
            frame = cv2.rectangle(frame,(50,top_left+360),(200,top_left+415), colours["HUD"],1)
            frame = cv2.line(frame,(125,top_left+415), (125,top_left+465), colours["HUD"],1)
            frame = cv2.line(frame,(125,top_left+465), (c_length-250-25,top_left+465), colours["HUD"],1)
            frame = cv2.circle(frame,(c_length-250,c_width+250),25, colours["HUD"],1)
        #cv2.imshow('fdfdg',frame)
        cv2.imshow('frame',frame)
        out.write(frame)
        frame_num += 1
        fps = frame_num / (time.time()-start)
    # Press Q on keyboard to exit 
        if cv2.waitKey(20) & 0xFF == ord('q'): 
            break

    # Break the loop 
        #else: 
        #    break
    
    # When everything done, release 
    # the video capture object 
    #cap.release() 
    out.release()
    # Closes all the frames 
    cv2.destroyAllWindows() 

#Send gcode to printer
def send_gcode(code):
    ser = serial.Serial(f"{PATH}/printer_data/comms/klippy.serial",115200)
    tx = f"{code} \n"
    ser.write(bytearray(tx.encode('utf-8')))
    ser.flush()

def printer_operation():
    global extrusion_factor, print_start
    preamble = ['M107',
                'G21',
                'G90',
                'M220 S40',
                'M204 S800',
                'M83',
                'G92 E0.0',
                'G1 E0.2 F150'
                ]
    
    code = """G1 F900
G1 X64 Y63.8
G1 X71 Y63.8 E2.75534
G1 X71 Y71 E2.75534
G1 X64 Y71 E2.75534
G1 X64 Y63.8 E2.73232

G1 X64 Y64.193

G1 X65 Y64.193 E0.2 F150
G1 X66 Y64.193 E0.2
G1 X67 Y64.193 E0.2
G1 X68 Y64.193 E0.2
G1 X69 Y64.193 E0.2
G1 X70 Y64.193 E0.2
G1 X71 Y64.193 E0.2 
G1 X71 Y64.57

G1 X70 Y64.57 E0.2
G1 X69 Y64.57 E0.2
G1 X68 Y64.57 E0.2
G1 X67 Y64.57 E0.2
G1 X66 Y64.57 E0.2
G1 X65 Y64.57 E0.2
G1 X64 Y64.57 E0.2
G1 X64 Y64.947

G1 X65 Y64.947 E0.2
G1 X66 Y64.947 E0.2
G1 X67 Y64.947 E0.2
G1 X68 Y64.947 E0.2
G1 X69 Y64.947 E0.2
G1 X70 Y64.947 E0.2
G1 X71 Y64.947 E0.2 
G1 X71 Y65.324 

G1 X70 Y65.324 E0.2
G1 X69 Y65.324 E0.2
G1 X68 Y65.324 E0.2
G1 X67 Y65.324 E0.2
G1 X66 Y65.324 E0.2
G1 X65 Y65.324 E0.2
G1 X64 Y65.324 E0.2
G1 X64 Y65.701 

G1 X65 Y65.701 E0.2
G1 X66 Y65.701 E0.2
G1 X67 Y65.701 E0.2
G1 X68 Y65.701 E0.2
G1 X69 Y65.701 E0.2
G1 X70 Y65.701 E0.2
G1 X71 Y65.701 E0.2
G1 X71 Y66.078 

G1 X70 Y66.078 E0.2
G1 X69 Y66.078 E0.2
G1 X68 Y66.078 E0.2
G1 X67 Y66.078 E0.2
G1 X66 Y66.078 E0.2
G1 X65 Y66.078 E0.2
G1 X64 Y66.078 E0.2
G1 X64 Y66.455 

G1 X65 Y66.455 E0.2
G1 X66 Y66.455 E0.2
G1 X67 Y66.455 E0.2
G1 X68 Y66.455 E0.2
G1 X69 Y66.455 E0.2
G1 X70 Y66.455 E0.2
G1 X71 Y66.455 E0.2
G1 X71 Y66.832 

G1 X70 Y66.832 E0.2
G1 X69 Y66.832 E0.2
G1 X68 Y66.832 E0.2
G1 X67 Y66.832 E0.2
G1 X66 Y66.832 E0.2
G1 X65 Y66.832 E0.2
G1 X64 Y66.832 E0.2
G1 X64 Y67.209 

G1 X65 Y67.209 E0.2
G1 X66 Y67.209 E0.2
G1 X67 Y67.209 E0.2
G1 X68 Y67.209 E0.2
G1 X69 Y67.209 E0.2
G1 X70 Y67.209 E0.2
G1 X71 Y67.209 E0.2
G1 X71 Y67.587

G1 X70 Y67.587 E0.2
G1 X69 Y67.587 E0.2
G1 X68 Y67.587 E0.2
G1 X67 Y67.587 E0.2
G1 X66 Y67.587 E0.2
G1 X65 Y67.587 E0.2
G1 X64 Y67.587 E0.2
G1 X64 Y67.964 

G1 X65 Y67.964 E0.2
G1 X66 Y67.964 E0.2
G1 X67 Y67.964 E0.2
G1 X68 Y67.964 E0.2
G1 X69 Y67.964 E0.2
G1 X70 Y67.964 E0.2
G1 X71 Y67.964 E0.2
G1 X71 Y68.341

G1 X70 Y68.341 E0.2
G1 X69 Y68.341 E0.2
G1 X68 Y68.341 E0.2
G1 X67 Y68.341 E0.2
G1 X66 Y68.341 E0.2
G1 X65 Y68.341 E0.2
G1 X64 Y68.341 E0.2
G1 X64 Y68.718

G1 X65 Y68.718 E0.2
G1 X66 Y68.718 E0.2
G1 X67 Y68.718 E0.2
G1 X68 Y68.718 E0.2
G1 X69 Y68.718 E0.2
G1 X70 Y68.718 E0.2
G1 X71 Y68.718 E0.2
G1 X71 Y69.095

G1 X70 Y69.095 E0.2
G1 X69 Y69.095 E0.2
G1 X68 Y69.095 E0.2
G1 X67 Y69.095 E0.2
G1 X66 Y69.095 E0.2
G1 X65 Y69.095 E0.2
G1 X64 Y69.095 E0.2
G1 X64 Y69.472

G1 X65 Y69.472 E0.2
G1 X66 Y69.472 E0.2
G1 X67 Y69.472 E0.2
G1 X68 Y69.472 E0.2
G1 X69 Y69.472 E0.2
G1 X70 Y69.472 E0.2
G1 X71 Y69.472 E0.2
G1 X71 Y69.849

G1 X70 Y69.849 E0.2
G1 X69 Y69.849 E0.2
G1 X68 Y69.849 E0.2
G1 X67 Y69.849 E0.2
G1 X66 Y69.849 E0.2
G1 X65 Y69.849 E0.2
G1 X64 Y69.849 E0.2
G1 X64 Y70.226 

G1 X65 Y70.226 E0.2
G1 X66 Y70.226 E0.2
G1 X67 Y70.226 E0.2
G1 X68 Y70.226 E0.2
G1 X69 Y70.226 E0.2
G1 X70 Y70.226 E0.2
G1 X71 Y70.226 E0.2
G1 X71 Y70.603 

G1 X70 Y70.603 E0.2
G1 X69 Y70.603 E0.2
G1 X68 Y70.603 E0.2
G1 X67 Y70.603 E0.2
G1 X66 Y70.603 E0.2
G1 X65 Y70.603 E0.2
G1 X64 Y70.603 E0.2

"""

    code = code.split("\n")
    gcode = []
    for c in code:
        if "G1" in c:
            gcode.append(c)

            

    
    for p in preamble:
        send_gcode(p)
        time.sleep(0.2)
    time.sleep(1)
    
    for idx,g in enumerate(gcode):
        e_val=0
        g_tx = ''
        for gx in g.split(' '):
            if 'E' in gx:
                e_val = float(gx.split('E')[-1])
                if idx > 4:
                    print_start= True
                    #e_val = e_val * extrusion_factor
                g_tx += f'E{e_val:.4f} '
            else:
                g_tx += f'{gx} '
        send_gcode(g_tx)
        time.sleep(1.0)
        

t1 = Thread(target=camera_system)
t2 = Thread(target=printer_operation)

t1.start()
time.sleep(5)
t2.start()