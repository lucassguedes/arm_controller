from mediapipe import solutions 
from mediapipe.framework.formats import landmark_pb2 
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision 


def draw_landmarks_on_image(rgb_image, detection_result):
	pose_landmarks_list = detection_result.pose_landmarks
	annotated_image = np.copy(rgb_image)

	# Loop through the detected poses to visualize
	for idx in range(len(pose_landmarks_list)): 
		pose_landmarks = pose_landmarks_list[idx]

		# Draw the pose landmarks
		pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
		pose_landmarks_proto.landmark.extend([
			landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
		])

		solutions.drawing_utils.draw_landmarks(
			annotated_image,
			pose_landmarks_proto,
			solutions.pose.POSE_CONNECTIONS,
			solutions.drawing_styles.get_default_pose_landmarks_style()
		)

		return annotated_image;


def pose_from_image():
	base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
	options = vision.PoseLandmarkerOptions(
		base_options=base_options,
		output_segmentation_masks=True
	)

	detector = vision.PoseLandmarker.create_from_options(options)

	image = mp.Image.create_from_file("girl-4051811_960_720.jpg")
	
	detection_result = detector.detect(image)

	annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

	cv2.imshow('Image', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
	cv2.waitKey(0)
	cv2.destroyAllWindows() # It destroys the image showing the window.

def main():
	cap = cv2.VideoCapture(0)

	mp_pose = mp.solutions.pose
	pose = mp_pose.Pose()
	mp_drawing = mp.solutions.drawing_utils

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			print("Ignoring empty camera frame.")
			continue

		# Convert the BGR image to RGB before processing
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# To improve performance
		rgb_frame.flags.writeable = False

		# Make detection
		results = pose.process(rgb_frame)

		# Draw the pose annotation on the image
		rgb_frame.flags.writeable = True
		frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

		if results.pose_landmarks:
			mp_drawing.draw_landmarks(
				frame,
				results.pose_landmarks,
				mp_pose.POSE_CONNECTIONS,
				mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
				mp_drawing.DrawingSpec(color=(66, 245, 96), thickness=2, circle_radius=2),
			)

		# Show the output
		cv2.imshow('MediaPipe Pose', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


if __name__ == '__main__':
	main()
	# pose_from_image()
