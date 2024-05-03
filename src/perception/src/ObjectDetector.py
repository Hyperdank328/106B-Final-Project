#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)

        self.bridge = CvBridge()

        self.cv_color_image = None

        self.color_image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.color_image_callback)

        self.image_pub = rospy.Publisher('detected_object', Image, queue_size=10)

        self.height_pub = rospy.Publisher('liquid_height', Float32, queue_size=10)
        rospy.spin()

    def color_image_callback(self, msg):
        try:
            self.cv_color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_images()
        except Exception as e:
            rospy.loginfo("Error in converting image: %s", e)

    def process_images(self):
        hsv = cv2.cvtColor(self.cv_color_image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Calculate the bounding rectangle of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            min_y = min(largest_contour, key=lambda item: item[0][1])[0][1]
            max_y = max(largest_contour, key=lambda item: item[0][1])[0][1]
            liquid_height = max_y - min_y

            self.height_pub.publish(liquid_height)

            rospy.loginfo("Height of the liquid in pixels: %f", liquid_height)
            # Draw the bounding rectangle around the detected liquid
            cv2.rectangle(self.cv_color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(self.cv_color_image, f"Liquid Height: {liquid_height} px", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ros_image = self.bridge.cv2_to_imgmsg(self.cv_color_image, "bgr8")
        self.image_pub.publish(ros_image)

if __name__ == '__main__':
    ObjectDetector()
