from mediapipe import solutions 
from mediapipe.framework.formats import landmark_pb2 
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision 
import serial


# Angle calculation function
def calculate_angle(a, b, c):
    """Calculate angle between three points (in degrees)."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point (joint)
    c = np.array(c)  # End point

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


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

		return annotated_image


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
	# Open the serial port
	ser = serial.Serial(port='COM5', baudrate=9600, timeout=1)
	cap = cv2.VideoCapture(0)

	mp_pose = mp.solutions.pose
	pose = mp_pose.Pose()
	mp_drawing = mp.solutions.drawing_utils

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			continue

		# Convert to RGB
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		rgb_frame.flags.writeable = False

		# Pose estimation
		results = pose.process(rgb_frame)
		rgb_frame.flags.writeable = True
		frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

		if results.pose_landmarks:
			landmarks = results.pose_landmarks.landmark

			# Get image dimensions
			h, w, _ = frame.shape

			# Helper to convert landmark to pixel
			def get_coords(index):
				lm = landmarks[index]
				return int(lm.x * w), int(lm.y * h)

			# Left arm: shoulder, elbow, wrist
			left_shoulder = get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
			left_elbow = get_coords(mp_pose.PoseLandmark.LEFT_ELBOW.value)
			left_wrist = get_coords(mp_pose.PoseLandmark.LEFT_WRIST.value)

			left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

			# Draw left angle
			cv2.putText(frame, str(int(left_angle)),
						(left_elbow[0] + 10, left_elbow[1] - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

			# Right arm: shoulder, elbow, wrist
			right_shoulder = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
			right_elbow = get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
			right_wrist = get_coords(mp_pose.PoseLandmark.RIGHT_WRIST.value)

			right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)


			if(right_angle > 90):
				right_angle = 0.0

			right_angle = int(right_angle)

			ser.write(f"{int(right_angle)}\n".encode('ascii'))
			print(right_angle)



			# Draw right angle
			cv2.putText(frame, str(int(right_angle)),
						(right_elbow[0] + 10, right_elbow[1] - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

			# Draw pose landmarks
			mp_drawing.draw_landmarks(
				frame,
				results.pose_landmarks,
				mp_pose.POSE_CONNECTIONS,
				mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
				mp_drawing.DrawingSpec(color=(66, 245, 96), thickness=2, circle_radius=2),
			)

		# Show the frame
		cv2.imshow('Pose Estimation with Angles', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	ser.close()


if __name__ == '__main__':
	main()
	# pose_from_image()
