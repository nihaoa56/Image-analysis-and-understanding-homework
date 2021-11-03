import cv2
import imageio

def read_video(video_path):
	video_cap = cv2.VideoCapture(video_path)
	frame_count = 0
	all_frames = []
	while True:
		ret, frame = video_cap.read()
		if ret is False:
			break
		frame = frame[..., ::-1]   # opencv读取BGR，转成RGB
		all_frames.append(frame)
		cv2.imshow('frame', frame)
		cv2.waitKey(1)
		frame_count += 1
		print(frame_count)
	video_cap.release()
	cv2.destroyAllWindows()
	print('===>', len(all_frames))

	return all_frames


def frame_to_gif(frame_list):
	gif = imageio.mimsave('./result.gif', frame_list, 'GIF', duration=0.00005)  
	# duration 表示图片间隔


if __name__ == "__main__":
	frame_list = read_video('input1.mp4')
	frame_to_gif(frame_list)
