# 导入包
import cv2
# 实例化一个对象
cap = cv2.VideoCapture('testvideo.mp4')
# 定义变量i
i = 0
dir = 'cvtoOut/'
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
