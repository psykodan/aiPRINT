import cv2
import numpy as np
import os
import shutil
from pathlib import Path


out1 = "synth_infill/test/"
out2 = "synth_line/test/"

os.makedirs(f"{out1}/good") 
os.makedirs(f"{out1}/under") 
os.makedirs(f"{out1}/over") 
os.makedirs(f"{out2}/good") 
os.makedirs(f"{out2}/under") 
os.makedirs(f"{out2}/over") 

#print_types = ["Concentric Plane_slice_1", "Rectilinear Plane_slice_1", "Spiral_slice_1"]
print_types = ["new_synth/"]
#print_runs = ["1","2","3","4"]
print_runs = []
for i in range(81,101,1):
    print_runs.append(str(i))
labels = ["over","under","good"]
limit = 758

for l in labels:
    img_num = 0
    for print_type in print_types:
        for print_run in print_runs:
            dirIn = f"{print_type}{print_run}"
            dirFiles = Path(dirIn).glob(f'{l}/*.jpg')
            for file in dirFiles:
                file_num = int(str(file).split("/")[-1].split(".")[0])

                if  file_num >= 758 or file_num<=315:

                    
                    if "over" in l:
                        shutil.copy(file, f"{out2}/over/{img_num}.jpg")
                        img_num += 1
                    elif "under" in l:
                        shutil.copy(file, f"{out2}/under/{img_num}.jpg")
                        img_num += 1
                    else:
                        shutil.copy(file, f"{out2}/{l}/{img_num}.jpg")
                        img_num += 1
                
                else:
                    if "over" in l:
                        shutil.copy(file, f"{out1}/over/{img_num}.jpg")
                        img_num += 1
                    elif "under" in l:
                        shutil.copy(file, f"{out1}/under/{img_num}.jpg")
                        img_num += 1
                    else:
                        shutil.copy(file, f"{out1}/{l}/{img_num}.jpg")
                        img_num += 1
                        