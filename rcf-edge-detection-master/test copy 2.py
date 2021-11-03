import torch
import models
import os
import numpy as np
from data_loader import BSDS_RCFLoader
from torch.utils.data import DataLoader
from PIL import Image
import scipy.io as io
import cv2

resume = 'ckpt/lr-0.01-iter-490000.pth'#训练结果参数
folder = 'result/val/'
batch_size = 1
assert batch_size == 1
#opencv图像视频切分
cap = cv2.VideoCapture('testvideo.mp4')
i = 0
dir = 'cvtoOut/'#保存位置·
# 先提取一张图片
success, frame = cap.read()
# 如果一张图片提取成功，则继续提取剩下的，直到视频结束
while success:
	# 将上面提取到的图片写入到文件夹
    frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(dir+str(i)+'.jpg', frame)
    # 更新变量i
    i += 1
    # 继续提取
    success, frame = cap.read()
#声明模型
model = models.resnet101(pretrained=False).cuda()
model.eval()

#读取训练结果参数
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint)
#读取数据
test_dataset = RCFLoader(split="test")
test_loader = DataLoader(
    test_dataset, batch_size=batch_size,
    num_workers=0, drop_last=False, shuffle=False)

if __name__ == "__main__":
    with torch.no_grad():
        for i, (image, ori, img_files) in enumerate(test_loader):
            h, w = ori.size()[2:]
            image = image.cuda()
            name = img_files[0][5:-4]
            outs = model(image, (h, w))
            fuse = outs[-1].squeeze().detach().cpu().numpy()
            outs.append(ori)
            print('working on .. {}'.format(i))
            #结果输出保存
            for result in outs:
                idx += 1
                result = result.squeeze().detach().cpu().numpy()
            Image.fromarray((fuse * 255).astype(np.uint8)).save('cvtoOut1/'+'{}.jpg'.format(name))
        print('finished.')

#生成视频
# 读取文件夹的路径，这里处理多个文件夹
IMG_DIR_1 = 'cvtoOut1/ut/'   
out_put_video_name = 'crfOutput.mp4' #输出视频的名称，
IMG_DIR = [IMG_DIR_1]
file_type = '.jpg' #图片后缀名为 .jpg，可以调整
fps = 10    #视频的FPS，可以适当调整
size=(768,432) # 输出视频的分辨率，这个需要和输入的图片集大小一致
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')#MP4格式输出
for dir_num in range(0,len(IMG_DIR)):
    IMG_DIR_temp = IMG_DIR[dir_num]
    out_video_file_name = out_put_video_name #设置视频的输出路径
    if os.path.exists(out_video_file_name):
    	os.remove(out_video_file_name)
    videoWriter = cv2.VideoWriter(out_video_file_name,fourcc,fps,size)#最后一个是保存图片的尺寸
    #生成保存图片集
    image_list = [f for f in os.listdir(IMG_DIR_temp) if f.endswith(file_type)]
    image_list.sort(key= lambda x:int(x[:-4]))
    for i in range(0, len(image_list)):
        frame = cv2.imread(IMG_DIR_temp+image_list[i])
        # 保存到视频中
        videoWriter.write(frame)
    videoWriter.release()

