import sys
import time
import cv2
cap = cv2.VideoCapture(0)
if cv2.cuda.getCudaEnabledDeviceCount():
	img_gpu_src = cv2.cuda_GpuMat()
	img_gpu_dst = cv2.cuda_GpuMat()
	#print(img_gpu_src)
else:
	print("cpu")
### Run with CPU
time_start = time.time()
while True:
	bl,img_src = cap.read()
	img_gpu_src.upload(img_src)
	img_gpu_dst = cv2.cuda.resize(img_gpu_src, (300, 300))
	#img_gpu_dst = cv2.cuda.cvtColor(img_gpu_src,cv2.COLOR_BGR2GRAY)
	img_dst = img_gpu_dst.download()
	cv2.imshow('GPU', img_dst)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
time_end = time.time()
print(time_end-time_end)
