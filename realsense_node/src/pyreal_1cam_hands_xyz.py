#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import datetime as dt

RENDER = True




class realsense_process:
    def __init__(self):
        self.pub_distance = rospy.Publisher('/hand', Float32MultiArray, queue_size=1)
        self.hands_msg = Float32MultiArray()
        self.left_hand = np.zeros(3)
        self.right_hand = np.zeros(3)

    def pub(self):
        self.pub_distance.publish(self.hands_msg)


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

                if RENDER == True:
                    images = cv2.putText(images, f"{hand_side} Hand xyz : ({realx:0.3}, {realy:0.3}, {realz:0.3})", org2, font, fontScale, color, thickness, cv2.LINE_AA)
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
        # Publish hands
        realsense_p.pub()


        # Display FPS
        time_diff = dt.datetime.today().timestamp() - start_time
        fps = int(1 / time_diff)
        org3 = (20, org[1] + 60)
        images = cv2.putText(images, f"FPS: {fps}", org3, font, fontScale, color, thickness, cv2.LINE_AA)
        rospy.loginfo(f"FPS: {fps}")

        # Display images 

        if RENDER == True:
            name_of_window = 'SN: ' + str(device)
            
            cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(name_of_window, images)
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

