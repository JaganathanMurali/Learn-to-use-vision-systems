#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSubscriber:
    def __init__(self):
        rospy.init_node('image_subscriber', anonymous=True)
        self.bridge = CvBridge()
        rospy.Subscriber('image_topic', Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            height, width = cv_image.shape[:2]  # Get height and width of the image

            # Define the desired width and height for resizing
            desired_width = 320
            desired_height = 240

            # Resize the image
            resized_image = cv2.resize(cv_image, (desired_width, desired_height))

            cv2.imshow("Resized Image", resized_image)
            cv2.waitKey(1)
        except Exception as e:
            print(e)

def main():
    try:
        image_subscriber = ImageSubscriber()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()

