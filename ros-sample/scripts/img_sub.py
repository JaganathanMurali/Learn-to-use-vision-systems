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
            # Define the rotation angle (in degrees)
            angle = 180
            # Calculate the rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            # Perform the rotation
            rotated_image = cv2.warpAffine(cv_image, rotation_matrix, (width, height))
            cv2.imshow("Rotated Image", rotated_image)
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

