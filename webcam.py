from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.dataloader import make_datapath_list, DataTransform
from utils.pspnet import PSPNet
import time
import cv2
#point = False
# ファイルパスリスト作成
rootpath = "./data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath=rootpath)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:",device)
net = PSPNet(n_classes=21).to(device)

# 学習済みパラメータをロード
state_dict = torch.load("./weights/pspnet50_150.pth",
                        map_location={'cuda:0': 'cpu'})
net.load_state_dict(state_dict)
 # 2. 前処理クラスの作成
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
transform = DataTransform(input_size=475, color_mean=color_mean, color_std=color_std)
anno_file_path = val_anno_list[0]
anno_class_img = Image.open(anno_file_path)   # [高さ][幅]
p_palette = anno_class_img.getpalette()
pp_palette = np.array(p_palette)
#print(type(p_palette))
#print("p_palette:{}".format(type(p_palette)))
#print(p_palette)
cv_palette = np.array(p_palette).reshape(-1,3)
phase = "val"
net.eval()

out_class_img = np.zeros((475,475,3))
cap = cv2.VideoCapture(0)
print(cv2.cuda.getCudaEnabledDeviceCount())
if(cap.isOpened() and cv2.cuda.getCudaEnabledDeviceCount()):
	print("Gpu")
else:
	print("No Gpu")
	exit(0)
img_gpu_src = cv2.cuda_GpuMat()
img_gpu_dst = cv2.cuda_GpuMat()
cnt = 0

while True:
	cnt += 1
	start1 = time.time()

	img,frame = cap.read()
	img_width, img_height,channel = frame.shape
	img_gpu_src.upload(frame)
	#色順を変えてPillow型にしていたがかえずにtransformをどうにかできればよき
	img_gpu_dst = cv2.cuda.resize(img_gpu_src,(475,475))
	frame = img_gpu_dst.download()
	img_gpu_dst = cv2.cuda.cvtColor(img_gpu_src,cv2.COLOR_BGR2RGB)
	cvimg = img_gpu_dst.download()
	imgobj = Image.fromarray(cvimg)
	img, anno_class_out = transform(phase, imgobj, anno_class_img)
	start2 = time.time()

	x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 475, 475])
	x = x.to(device)
	outputs = net(x)
	start25 = time.time()

	y = outputs[0]  # AuxLoss側は無視 yのサイズはtorch.Size([1, 21, 475, 475])
	y = y.to("cpu")
	y = y[0].detach().numpy()  # y：torch.Size([1, 21, 475, 475])
	y = np.argmax(y, axis=0)
	start3 = time.time()
	out_class_img = cv_palette[y]
	result = cv2.cuda.addWeighted(src1=out_class_img,alpha=0.5,src2=frame,beta=0.8,gamma=0,dtype=cv2.CV_8U)
	cv2.imshow('GPU',result)
	start4 = time.time()
	if not cnt%20:
		print ("elapsed_time2-1:{0}".format(start2-start1) + "[sec]")
		print ("elapsed_time25-2:{0}".format(start25-start2) + "[sec]")
		print ("elapsed_time3-25:{0}".format(start3-start25) + "[sec]")
		print ("elapsed_time4-3:{0}".format(start4-start3) + "[sec]")
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
