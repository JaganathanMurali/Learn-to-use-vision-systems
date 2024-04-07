#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageDifferenceSubscriber:
    def __init__(self):
        rospy.init_node('image_difference_subscriber', anonymous=True)
        self.bridge = CvBridge()

        # Subscribe to the two image topics
        rospy.Subscriber('image_topic1', Image, self.callback_image1)
        rospy.Subscriber('image_topic2', Image, self.callback_image2)

    def callback_image1(self, data):
        try:
            self.image1 = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except Exception as e:
            print(e)

    def callback_image2(self, data):
        try:
            self.image2 = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

            if hasattr(self, 'image1'):  # Ensure image1 has been received
                # Compute the absolute difference between image1 and image2
                diff_image = cv2.absdiff(self.image1, self.image2)

                # Display the difference image
                cv2.imshow("Image Difference", diff_image)
                cv2.waitKey(1)
        except Exception as e:
            print(e)

def main():
    try:
        image_diff_subscriber = ImageDifferenceSubscriber()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()

