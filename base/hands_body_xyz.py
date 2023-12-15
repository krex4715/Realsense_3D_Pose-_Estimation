# ====== Sample Code for Smart Design Technology Blog ======

# Intel Realsense cam has RGB camera with 1920×1080 resolution
# Depth camera is 1280x720
# FOV is limited to 69deg x 42deg (H x V) - the RGB camera FOV

# If you run this on a non-Intel CPU, explore other options for rs.align
    # On the NVIDIA Jetson AGX we build the pyrealsense lib with CUDA

import pyrealsense2 as rs
import mediapipe as mp
import cv2
cv2.namedWindow('realsense_yolo', cv2.WINDOW_AUTOSIZE)
import numpy as np
import datetime as dt
from ultralytics import YOLO
from util import get_most_min_distance, draw_skeleton

PAIRS = [
    (5, 6),  # shoulder
    (5, 11), (11, 12), (12, 6),  # body
    # (11, 15), (11, 13), (13, 15),  # left leg
    # (12, 16), (12, 14), (14, 16),  # right leg
    (5, 9), (9, 7), (7, 5),  # left arm
    (6, 10), (10, 8), (8, 6)  # right arm
]



font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 20)
fontScale = .5
color = (0,50,255)
thickness = 1

# ====== Realsense ======
realsense_ctx = rs.context()
connected_devices = [] # List of serial numbers for present cameras
for i in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
    print(f"{detected_camera}")
    connected_devices.append(detected_camera)
device = connected_devices[0] # In this example we are only using one camera
pipeline = rs.pipeline()
config = rs.config()
background_removed_color = 153 # Grey

# ====== Mediapipe ======
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


# ====== YOLO Model ======
model = YOLO('yolov8n-pose.pt')


# ====== Enable Streams ======
config.enable_device(device)

# # For worse FPS, but better resolution:
# stream_res_x = 1280
# stream_res_y = 720
# # For better FPS. but worse resolution:
stream_res_x = 640
stream_res_y = 480

stream_fps = 30

config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

# ====== Get depth Scale ======
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"\tDepth Scale for Camera SN {device} is: {depth_scale}")

# ====== Set clipping distance ======
clipping_distance_in_meters = 2
clipping_distance = clipping_distance_in_meters / depth_scale
print(f"\tConfiguration Successful for SN {device}")

# ====== Get and process images ====== 
print(f"Starting to capture images on SN: {device}")

while True:
    start_time = dt.datetime.today().timestamp()

    # Get and align frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    intr = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    color_frame = aligned_frames.get_color_frame()
    
    if not aligned_depth_frame or not color_frame:
        continue

    # Process images
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_image_flipped = cv2.flip(depth_image,1)
    color_image = np.asanyarray(color_frame.get_data())

    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #Depth image is 1 channel, while color image is 3
    background_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), background_removed_color, color_image)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    images = cv2.flip(background_removed,1)
    color_image = cv2.flip(color_image,1)
    color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Process hands
    results = hands.process(color_images_rgb)
    if results.multi_hand_landmarks:
        number_of_hands = len(results.multi_hand_landmarks)
        i=0
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(images, handLms, mpHands.HAND_CONNECTIONS)
            org2 = (20, org[1]+(20*(i+1)))
            hand_side_classification_list = results.multi_handedness[i]
            hand_side = hand_side_classification_list.classification[0].label
            # ====== Get middle finger knuckle ======
            middle_finger_knuckle = results.multi_hand_landmarks[i].landmark[9]
            x = int(middle_finger_knuckle.x*len(depth_image_flipped[0]))
            y = int(middle_finger_knuckle.y*len(depth_image_flipped))
            
            if x >= len(depth_image_flipped[0]):
                x = len(depth_image_flipped[0]) - 1
            if y >= len(depth_image_flipped):
                y = len(depth_image_flipped) - 1
            # 중간 손가락 마디의 깊이 값 가져오기
            z = depth_image_flipped[y, x] * depth_scale
                
            mfk_distance = depth_image_flipped[y,x] * depth_scale # meters
            # mfk_distance_feet = mfk_distance * 3.281 # feet

            realx, realy, realz = rs.rs2_deproject_pixel_to_point(intr, [x, y], z)
            realy = -realy
            print(f"realx: {realx}, realy: {realy}, realz: {realz}")
            # images = cv2.putText(images, f"{hand_side} Hand Distance: ({mfk_distance:0.3} m) away", org2, font, fontScale, color, thickness, cv2.LINE_AA)
            # put realx, realy, realz value on image
            images = cv2.putText(images, f"{hand_side} Hand xyz : ({realx:0.3}, {realy:0.3}, {realz:0.3})", org2, font, fontScale, color, thickness, cv2.LINE_AA)
           
            i+=1
        images = cv2.putText(images, f"Hands: {number_of_hands}", org, font, fontScale, color, thickness, cv2.LINE_AA)
    else:
        images = cv2.putText(images,"No Hands", org, font, fontScale, color, thickness, cv2.LINE_AA)

    # YOLO
    yolo_results = model(color_image)
    frame_center = np.array([color_image.shape[1] / 2, color_image.shape[0] / 2])


    # if video, length of results is 1
    keypoints_xy = yolo_results[0].keypoints.cpu().xy.squeeze().numpy().tolist()
    keypoints_xyn = yolo_results[0].keypoints.cpu().xyn.squeeze().numpy().tolist()
    boxes = yolo_results[0].boxes.xywh.cpu().numpy()

    box_center = boxes[:, :2]

    min_idx = get_most_min_distance(box_center, frame_center)
    if min_idx == -1:  # 사람 객체가 없으면 skip
        continue

    if np.array(keypoints_xy).shape == (17, 2): # 사람이 한명
        closest_keypoints = np.array(keypoints_xy)
    else: # 사람이 여러명
        closest_keypoints = np.array(keypoints_xy)[min_idx] # 중앙에 있는 사람
    # depth average
    radius = 15
    # annotated_frame = draw_skeleton(images, closest_keypoints, PAIRS, radius=radius)


    ## 머리 x,y,z계산 closest_keypoints
    head_x, head_y, head_z = rs.rs2_deproject_pixel_to_point(intr, [int(closest_keypoints[0][0]), int(closest_keypoints[0][1])], depth_image_flipped[int(closest_keypoints[0][1]), int(closest_keypoints[0][0])] * depth_scale)
    head_y = -head_y
    print(f"head_x: {head_x}, head_y: {head_y}, head_z: {head_z}")
    images = cv2.putText(images, f"Head xyz : ({head_x:0.3}, {head_y:0.3}, {head_z:0.3})", (20,300), font, fontScale, color, thickness, cv2.LINE_AA)
    # 머리 중심점
    x, y = int(closest_keypoints[0][0]), int(closest_keypoints[0][1])
    cv2.circle(images, (x, y), 5, (0, 255, 0), 1)






    # ## 몸통중심 x,y,z계산  5,6,11,12 중심값
    # bodycenter_x = (closest_keypoints[5][0] + closest_keypoints[6][0] + closest_keypoints[11][0] + closest_keypoints[12][0]) / 4
    # bodycenter_y = (closest_keypoints[5][1] + closest_keypoints[6][1] + closest_keypoints[11][1] + closest_keypoints[12][1]) / 4
    # bodycenter_x, bodycenter_y, bodycenter_z = rs.rs2_deproject_pixel_to_point(intr, [int(bodycenter_x), int(bodycenter_y)], depth_image_flipped[int(bodycenter_y), int(bodycenter_x)] * depth_scale)
    # bodycenter_y = -bodycenter_y
    # print(f"bodycenter_x: {bodycenter_x}, bodycenter_y: {bodycenter_y}, bodycenter_z: {bodycenter_z}")
    # images = cv2.putText(images, f"Bodycenter xyz : ({bodycenter_x:0.3}, {bodycenter_y:0.3}, {bodycenter_z:0.3})", (20,320), font, fontScale, color, thickness, cv2.LINE_AA)

    # # 몸통점
    # for point in [5, 6, 11, 12]:
    #     x, y = int(closest_keypoints[point][0]), int(closest_keypoints[point][1])
    #     cv2.circle(images, (x, y), 5, (0, 255, 0), 1)

    

    




    # Display FPS
    time_diff = dt.datetime.today().timestamp() - start_time
    fps = int(1 / time_diff)
    org3 = (20, org[1] + 60)
    images = cv2.putText(images, f"FPS: {fps}", org3, font, fontScale, color, thickness, cv2.LINE_AA)

    # name_of_window = 'SN: ' + str(device)

    # Display images 
    
    cv2.imshow('realsense_yolo', images)
    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        print(f"User pressed break key for SN: {device}")
        break

print(f"Application Closing")
pipeline.stop()
print(f"Application Closed.")