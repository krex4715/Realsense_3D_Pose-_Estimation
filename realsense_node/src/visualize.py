#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
import numpy as np





class realsense_visualize:
    def __init__(self):
        self.sub_hand = rospy.Subscriber('/hand_xyz', Float32MultiArray, self.hand_callback)
        self.sub_body = rospy.Subscriber('/body_xyz', Float32MultiArray, self.body_callback)

        self.left_hand = None
        self.right_hand = None
        self.head = None
        self.leftshoulder = None
        self.leftelbow = None
        self.rightshoulder = None
        self.rightelbow = None

        self.pub_lefthand = rospy.Publisher('/lefthand_marker', Marker, queue_size=1)
        self.pub_righthand = rospy.Publisher('/righthand_marker', Marker, queue_size=1)
        self.pub_head = rospy.Publisher('/head_marker', Marker, queue_size=1)
        self.pub_leftshoulder = rospy.Publisher('/leftshoulder_marker', Marker, queue_size=1)
        self.pub_leftelbow = rospy.Publisher('/leftelbow_marker', Marker, queue_size=1)
        self.pub_rightshoulder = rospy.Publisher('/rightshoulder_marker', Marker, queue_size=1)
        self.pub_rightelbow = rospy.Publisher('/rightelbow_marker', Marker, queue_size=1)




    def hand_callback(self, msg):
        self.left_hand = msg.data[0:3]
        self.right_hand = msg.data[3:6]
        print(self.right_hand)

    def body_callback(self, msg):
        self.head = msg.data[0:3]
        self.leftshoulder = msg.data[3:6]
        self.leftelbow = msg.data[6:9]
        self.rightshoulder = msg.data[9:12]
        self.rightelbow = msg.data[12:15]

    def marker_publish(self,marker,data,frame_id,rgba,scale,publisher,origin=[0,0,0]):
        marker.header.frame_id = frame_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]
        marker.pose.position.x = data.x + origin[0]
        marker.pose.position.y = data.y + origin[1]
        marker.pose.position.z = data.z + origin[2]

        publisher.publish(marker)



def main():
    rospy.init_node('realsense_visualize')
    rv = realsense_visualize()
    rate = rospy.Rate(30)

    left_hand_marker = Marker()
    right_hand_marker = Marker()

    head_marker = Marker()
    leftshoulder_marker = Marker()
    leftelbow_marker = Marker()
    rightshoulder_marker = Marker()
    rightelbow_marker = Marker()

    ORIGIN = [0,0,0.5]
    while not rospy.is_shutdown():
        if rv.left_hand is not None and rv.left_hand != (0,0,0):
            rv.marker_publish(left_hand_marker,Point(rv.left_hand[2],rv.left_hand[0],rv.left_hand[1]),'map',[1,0,0.2,1],0.05,rv.pub_lefthand,ORIGIN)
        if rv.right_hand is not None and rv.right_hand != (0,0,0):
            rv.marker_publish(right_hand_marker,Point(rv.right_hand[2],rv.right_hand[0],rv.right_hand[1]),'map',[1,0,0.2,1],0.05,rv.pub_righthand,ORIGIN)
        if rv.head is not None and rv.head != (0,0,0):
            rv.marker_publish(head_marker,Point(rv.head[2],rv.head[0],rv.head[1]),'map',[1,0,0,1],0.05,rv.pub_head,ORIGIN)
        if rv.leftshoulder is not None and rv.leftshoulder != (0,0,0):
            rv.marker_publish(leftshoulder_marker,Point(rv.leftshoulder[2],rv.leftshoulder[0],rv.leftshoulder[1]),'map',[0,1,0,1],0.05,rv.pub_leftshoulder,ORIGIN)
        if rv.leftelbow is not None and rv.leftelbow != (0,0,0):
            rv.marker_publish(leftelbow_marker,Point(rv.leftelbow[2],rv.leftelbow[0],rv.leftelbow[1]),'map',[0,1,0,1],0.05,rv.pub_leftelbow,ORIGIN)
        if rv.rightshoulder is not None and rv.rightshoulder != (0,0,0):
            rv.marker_publish(rightshoulder_marker,Point(rv.rightshoulder[2],rv.rightshoulder[0],rv.rightshoulder[1]),'map',[0,1,0,1],0.05,rv.pub_rightshoulder,ORIGIN)
        if rv.rightelbow is not None and rv.rightelbow != (0,0,0):
            rv.marker_publish(rightelbow_marker,Point(rv.rightelbow[2],rv.rightelbow[0],rv.rightelbow[1]),'map',[0,1,0,1],0.05,rv.pub_rightelbow,ORIGIN)

        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

