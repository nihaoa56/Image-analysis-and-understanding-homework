import cv2
cap = cv2.VideoCapture("testvideo.mp4")  #打开视频
while (1):
    ret, frame = cap.read()  # 读取一帧视频
    # ret 读取了数据就返回True,没有读取数据(已到尾部)就返回False
    # frame 返回读取的视频数据--一帧数据
    print(frame)
    cv2.imshow("capture", frame)  # 显示视频帧

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 等候40ms,播放下一帧，或者按q键退出
        break
cap.release()  # 释放视频流
cv2.destroyAllWindows() # 关闭所有窗口