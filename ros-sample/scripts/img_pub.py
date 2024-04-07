#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class ImagePublisher:
    def __init__(self):
        rospy.init_node('image_publisher', anonymous=True)
        self.image_pub = rospy.Publisher('image_topic', Image, queue_size=10)
        self.bridge = CvBridge()
        self.rate = rospy.Rate(1)  # Publish rate of 1 Hz
        self.image_path = "/home/yuvanparker/Downloads/cat.jpg"  # Specify the path to your image

    def publish_image(self):
        while not rospy.is_shutdown():
            cv_image = cv2.imread(self.image_path)
            if cv_image is not None:
                ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
                self.image_pub.publish(ros_image)
                rospy.loginfo("Image published: {}".format(self.image_path))
                self.rate.sleep()

def main():
    try:
        image_publisher = ImagePublisher()
        image_publisher.publish_image()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()

