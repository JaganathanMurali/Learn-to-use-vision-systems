#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

def publish_images():
    rospy.init_node('image_publisher', anonymous=True)
    rate = rospy.Rate(1)  # Publish rate (1 Hz)

    # Create publishers for two image topics
    image_pub1 = rospy.Publisher('image_topic1', Image, queue_size=10)
    image_pub2 = rospy.Publisher('image_topic2', Image, queue_size=10)

    bridge = CvBridge()

    # Load two sample images (replace these with your actual images)
    image1 = cv2.imread('/home/yuvanparker/Downloads/cat.jpg')  # Load your first image
    image2 = cv2.imread('/home/yuvanparker/Downloads/cat.jpg')  # Load your second image

    while not rospy.is_shutdown():
        # Convert images to ROS Image messages
        image_msg1 = bridge.cv2_to_imgmsg(image1, encoding="bgr8")
        image_msg2 = bridge.cv2_to_imgmsg(image2, encoding="bgr8")

        # Publish the images to their respective topics
        image_pub1.publish(image_msg1)
        image_pub2.publish(image_msg2)

        rate.sleep()

if __name__ == '__main__':
    try:
        publish_images()
    except rospy.ROSInterruptException:
        pass

