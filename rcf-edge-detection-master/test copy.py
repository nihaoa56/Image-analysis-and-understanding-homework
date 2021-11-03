import torch
import models
import os
import numpy as np
from data_loader import BSDS_RCFLoader
from torch.utils.data import DataLoader
from PIL import Image
import scipy.io as io
import cv2

resume = 'ckpt/lr-0.01-iter-490000.pth'
folder = 'result/val/'
all_folder = os.path.join(folder, 'all/')
print(all_folder)
png_folder = os.path.join(folder, 'png/')
mat_folder = os.path.join(folder, 'mat/')
batch_size = 1
assert batch_size == 1
#opencv图像视频切分
cap = cv2.VideoCapture('testvideo.mp4')
# 定义变量i
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

# os.mkdir(all_folder)
# os.mkdir(png_folder)
# os.mkdir(mat_folder)
# except Exception:
#     print('dir already exist....')
#     pass

model = models.resnet101(pretrained=False).cuda()
model.eval()

#resume..
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint)

test_dataset = BSDS_RCFLoader(split="test")
# print(len(test_dataset))
test_loader = DataLoader(
    test_dataset, batch_size=batch_size,
    num_workers=0, drop_last=False, shuffle=False)

if __name__ == "__main__":
    with torch.no_grad():
        for i, (image, ori, img_files) in enumerate(test_loader):
            # print(img_files)
            h, w = ori.size()[2:]
            # print(h,w)
            # print()
            image = image.cuda()
            # print(image)
            name = img_files[0][5:-4]

            outs = model(image, (h, w))
            fuse = outs[-1].squeeze().detach().cpu().numpy()

            outs.append(ori)

            idx = 0
            print('working on .. {}'.format(i))

            for result in outs:
                idx += 1
                result = result.squeeze().detach().cpu().numpy()
                # if len(result.shape) == 3:
                #     result = result.transpose(1, 2, 0).astype(np.uint8)
                #     result = result[:, :, [2, 1, 0]]
                #     Image.fromarray(result).save('cvtoOut1/'+'{}.jpg'.format(idx))
                # else:
                #     result = (result * 255).astype(np.uint8)
                #     # result\val\all
                #     Image.fromarray(result).save('cvtoOut1/'+'{}.png'.format(idx))
            Image.fromarray((fuse * 255).astype(np.uint8)).save('cvtoOut1/'+'{}.jpg'.format(name))
            # io.savemat(os.path.join(mat_folder, '{}.mat'.format(name)), {'result': fuse})
        print('finished.')

#生成视频
# 读取文件夹的路径，这里处理多个文件夹
IMG_DIR_1 = 'cvtoOut1/ut/'   
# IMG_DIR_2 = '/home/xilinx/jupyter_notebooks/skynet_final/dataset_training/riding4/'
# IMG_DIR_3 = '/home/xilinx/jupyter_notebooks/skynet_final/dataset_training/car11/'
# IMG_DIR_4 = '/home/xilinx/jupyter_notebooks/skynet_final/dataset_training/truck1/'
out_put_video_name = 'crfOutput.mp4' #输出视频的名称，

IMG_DIR = [IMG_DIR_1]
file_type = '.jpg' #图片后缀名为 .jpg，可以调整
fps = 10    #视频的FPS，可以适当调整
size=(768,432) # 输出视频的分辨率
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

for dir_num in range(0,len(IMG_DIR)):
    IMG_DIR_temp = IMG_DIR[dir_num]
    out_video_file_name = out_put_video_name #设置视频的输出路径为 原文件夹下+前面设置的输出视频的名称
    if os.path.exists(out_video_file_name):
    	os.remove(out_video_file_name)
    videoWriter = cv2.VideoWriter(out_video_file_name,fourcc,fps,size)#最后一个是保存图片的尺寸
    image_list = [f for f in os.listdir(IMG_DIR_temp) if f.endswith(file_type)]
    # print(image_list)
    image_list.sort(key= lambda x:int(x[:-4]))
    print(image_list)
    for i in range(0, len(image_list)):
        # print(IMG_DIR_temp+image_list[i])
        frame = cv2.imread(IMG_DIR_temp+image_list[i])
        # print(frame)
        videoWriter.write(frame)
    videoWriter.release()

