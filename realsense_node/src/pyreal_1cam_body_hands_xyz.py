#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray


RENDER = True
import cv2
if RENDER == True:
    cv2.namedWindow('realsense yolo', cv2.WINDOW_AUTOSIZE)
import mediapipe as mp
import pyrealsense2 as rs
import datetime as dt
from ultralytics import YOLO
from util import get_most_min_distance, draw_skeleton,keypoint_to_xyz



PAIRS = [
    (5, 6),  # shoulder
    (5, 11), (11, 12), (12, 6),  # body
    # (11, 15), (11, 13), (13, 15),  # left leg
    # (12, 16), (12, 14), (14, 16),  # right leg
    (5, 9), (9, 7), (7, 5),  # left arm
    (6, 10), (10, 8), (8, 6)  # right arm
]



class realsense_process:
    def __init__(self):
        self.pub_hand_distance = rospy.Publisher('/hand_xyz', Float32MultiArray, queue_size=1)
        self.pub_body_distance = rospy.Publisher('/body_xyz', Float32MultiArray, queue_size=1)

        self.hands_msg = Float32MultiArray()
        self.left_hand = np.zeros(3)
        self.right_hand = np.zeros(3)

        self.body_msg = Float32MultiArray()
        self.body = np.zeros(5*3)

    def pub(self):
        self.pub_hand_distance.publish(self.hands_msg)
        self.pub_body_distance.publish(self.body_msg)


def main():

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (20, 100)
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
    hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.2, min_tracking_confidence=0.2)
    mpDraw = mp.solutions.drawing_utils

    # ====== YOLO ======
    yolo_model = YOLO('/home/krex/catkin_ws/src/realsense_node/src/yolov8n-pose.pt')




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
        


    # initialize node
    rospy.init_node('realsense_node', anonymous=False)
    rate = rospy.Rate(30)

    realsense_p = realsense_process()
    while not rospy.is_shutdown():
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
                index_finger_knuckle = results.multi_hand_landmarks[i].landmark[5]
                x = int(index_finger_knuckle.x*len(depth_image_flipped[0]))
                y = int(index_finger_knuckle.y*len(depth_image_flipped))
                
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

                if hand_side == 'Left':
                    realsense_p.left_hand = np.array([realx, realy, realz])
                elif hand_side == 'Right':
                    realsense_p.right_hand = np.array([realx, realy, realz])

                # if RENDER == True:
                #     images = cv2.putText(images, f"{hand_side} Hand xyz : ({realx:0.3}, {realy:0.3}, {realz:0.3})", org2, font, fontScale, color, thickness, cv2.LINE_AA)
                i+=1
                # print(i)
            
            if number_of_hands == 2:
                realsense_p.hands_msg.data = np.concatenate((realsense_p.left_hand, realsense_p.right_hand))
            elif number_of_hands == 1:
                if hand_side == 'Left':
                    realsense_p.hands_msg.data = np.concatenate((realsense_p.left_hand, np.zeros(3)))
                elif hand_side == 'Right':
                    realsense_p.hands_msg.data = np.concatenate((np.zeros(3), realsense_p.right_hand))

            
            if RENDER == True:
                images = cv2.putText(images, f"Hands: {number_of_hands}", org, font, fontScale, color, thickness, cv2.LINE_AA)
        else:
            realsense_p.hands_msg.data = np.zeros(6)
            if RENDER == True:
                images = cv2.putText(images,"No Hands", org, font, fontScale, color, thickness, cv2.LINE_AA)
    
    
        # YOLO
        yolo_results = yolo_model(color_image)
        frame_center = np.array([len(color_image[0])/2, len(color_image)/2])

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
        

        head_pixel, head = keypoint_to_xyz(closest_keypoints,0,depth_image_flipped,rs,intr,depth_scale)
        (head_pixel_x, head_pixel_y),(head_x, head_y, head_z) = head_pixel, head

        leftshoulder_pixel, left_shoulder = keypoint_to_xyz(closest_keypoints,5,depth_image_flipped,rs,intr,depth_scale)
        (leftshoulder_pixel_x, leftshoulder_pixel_y),(left_shoulder_x, left_shoulder_y, left_shoulder_z) = leftshoulder_pixel, left_shoulder

        leftelbow_pixel, left_elbow = keypoint_to_xyz(closest_keypoints,7,depth_image_flipped,rs,intr,depth_scale)
        (leftelbow_pixel_x, leftelbow_pixel_y),(left_elbow_x, left_elbow_y, left_elbow_z) = leftelbow_pixel, left_elbow

        rightshoulder_pixel, right_shoulder = keypoint_to_xyz(closest_keypoints,6,depth_image_flipped,rs,intr,depth_scale)
        (rightshoulder_pixel_x, rightshoulder_pixel_y),(right_shoulder_x, right_shoulder_y, right_shoulder_z) = rightshoulder_pixel, right_shoulder

        rightelbow_pixel, right_elbow = keypoint_to_xyz(closest_keypoints,8,depth_image_flipped,rs,intr,depth_scale)
        (rightelbow_pixel_x, rightelbow_pixel_y),(right_elbow_x, right_elbow_y, right_elbow_z) = rightelbow_pixel, right_elbow

        
        realsense_p.body_msg.data = np.array([head_x, head_y, head_z, 
                                            left_shoulder_x, left_shoulder_y, left_shoulder_z,
                                            left_elbow_x, left_elbow_y, left_elbow_z,
                                            right_shoulder_x, right_shoulder_y, right_shoulder_z,
                                            right_elbow_x, right_elbow_y, right_elbow_z])

        if RENDER == True:
            cv2.circle(images, (head_pixel_x, head_pixel_y), 5, (0, 0, 255), -1)  # red circle
            cv2.circle(images, (leftshoulder_pixel_x, leftshoulder_pixel_y), 5, (0, 255, 0), -1)
            cv2.circle(images, (leftelbow_pixel_x, leftelbow_pixel_y), 5, (0, 255, 0), -1)
            cv2.circle(images, (rightshoulder_pixel_x, rightshoulder_pixel_y), 5, (0, 255, 0), -1)
            cv2.circle(images, (rightelbow_pixel_x, rightelbow_pixel_y), 5, (0, 255, 0), -1)

        
        
        # Publish hands
        realsense_p.pub()














        # Display FPS
        time_diff = dt.datetime.today().timestamp() - start_time
        fps = int(1 / time_diff)
        if RENDER == True:
            org3 = (20, org[1] + 60)
            images = cv2.putText(images, f"FPS: {fps}", org3, font, fontScale, color, thickness, cv2.LINE_AA)
        rospy.loginfo(f"FPS: {fps}")

        # Display images 

        if RENDER == True:
            cv2.imshow('realsense yolo', images)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q') or key == 27:
                print(f"User pressed break key for SN: {device}")
                break

        
    

        rate.sleep()



if __name__ == '__main__':
    try:
        main()
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException:
        pass

