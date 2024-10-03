import cv2
vidcap = cv2.VideoCapture('Data_gather_070224/5.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("Data_gather_070224/frames/5/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
