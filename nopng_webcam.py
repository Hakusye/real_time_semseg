# OpenCV のインポート
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
net = PSPNet(n_classes=21)

# 学習済みパラメータをロード
state_dict = torch.load("./weights/pspnet50_30.pth",
                        map_location={'cuda:0': 'cpu'})
net.load_state_dict(state_dict)
 # 2. 前処理クラスの作成
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
transform = DataTransform(input_size=475, color_mean=color_mean, color_std=color_std)
anno_file_path = val_anno_list[0]
anno_class_img = Image.open(anno_file_path)   # [高さ][幅]
p_palette = anno_class_img.getpalette()
phase = "val"

cap = cv2.VideoCapture(0)
# 3. 前処理
#anno_file_path = val_anno_list[0]
#anno_class_img = Image.open(anno_file_path)   # [高さ][幅]
#p_palette = anno_class_img.getpalette()
#phase = "val"
while True:
# VideoCaptureから1フレーム読み込む
        img,frame = cap.read()
        #img = cap.read()
        #frame = cv2.resize(frame, (int(frame.shape[1]/4), int(frame.shape[0]/4)))
        #cv2.imshow('Raw Frame',frame)
        #処理
        #cv2.imwrite('web.jpg',frame)
        #image_file_path = "web.jpg"
        #img = Image.open(image_file_path)   # [高さ][幅][色RGB]
        #cv2.imshow('image',frame)
        cvimg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img_width, img_height,channel = cvimg.shape
        #image obj
        imgobj = Image.fromarray(cvimg)
        #img = Image.new('RGB',(img_width,img_height))
        #get color
        #img = np.array([[cvimg[i,j] for j in range(img_height)] for i in range(img_width)])
        #print(img.size)
    
      
        img, anno_class_img = transform(phase, imgobj, anno_class_img)

# 4. PSPNetで推論する
        net.eval()
        x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 475, 475])
        outputs = net(x)
        y = outputs[0]  # AuxLoss側は無視 yのサイズはtorch.Size([1, 21, 475, 475])
    # 5. PSPNetの出力から最大クラスを求め、カラーパレット形式にし、画像サイズを元に戻す
        y = y[0].detach().numpy()  # y：torch.Size([1, 21, 475, 475])
        y = np.argmax(y, axis=0)
        anno_class_img = Image.fromarray(np.uint8(y), mode="P")
        #anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
        anno_class_img.putpalette(p_palette)

    # 6. 画像を透過させて重ねる
        trans_img = Image.new('RGBA', anno_class_img.size, (0, 0, 0, 0))
        anno_class_img = anno_class_img.convert('RGBA')  # カラーパレット形式をRGBAに変換
        
        for x in range(475):
            for y in range(475):
                # 推論結果画像のピクセルデータを取得
                pixel = anno_class_img.getpixel((x, y))
                r, g, b, a = pixel

                # (0, 0, 0)の背景ならそのままにして透過させる
                if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                    continue
                else:
                    # それ以外の色は用意した画像にピクセルを書き込む
                    trans_img.putpixel((x, y), (r, g, b, 150))
                    # 150は透過度の大きさを指定している

        #img = Image.open(image_file_path)   # [高さ][幅][色RGB]
        imgobj = imgobj.resize((475,475),Image.NEAREST)
        result = Image.alpha_composite(imgobj.convert('RGBA'),trans_img)
        #result = cv2.cvtColor(np.float32(result),cv2.COLOR_RGBA2BGRA)
        #plt.imshow(anno_class_img)
        plt.imshow(result)
        #plt.show()
        plt.pause(0.01)
       
        #cv2.imshow('image',result)
        
        k = cv2.waitKey(1)
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
