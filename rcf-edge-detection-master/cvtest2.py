import cv2
import os

# 读取文件夹的路径，这里处理多个文件夹
IMG_DIR_1 = 'cvtoOut1/ut/'   
# IMG_DIR_2 = '/home/xilinx/jupyter_notebooks/skynet_final/dataset_training/riding4/'
# IMG_DIR_3 = '/home/xilinx/jupyter_notebooks/skynet_final/dataset_training/car11/'
# IMG_DIR_4 = '/home/xilinx/jupyter_notebooks/skynet_final/dataset_training/truck1/'
out_put_video_name = 'output.mp4' #输出视频的名称，

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