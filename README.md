## Realsesnse 3D Pose Estimation ROS Package


**_Copy & Paste 'realsense_node' directory to your workspace_**




### Only used YOLOv8n-pose (FPS : 18~20)
```bash
roslaunch realsense_node realsense_rviz_only_yolo.launch
```

![img](./img/yolo.gif)
![img](./img/yolo2.gif)




### Used YOLOv8n-pose & mediapipe (FPS : 8~10)
```bash
roslaunch realsense_node realsense_rviz.launch
```

![img](./img/mediapipe_yolo.gif)