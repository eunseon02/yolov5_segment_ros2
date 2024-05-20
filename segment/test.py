import os
import platform
import sys
from pathlib import Path

import torch
import numpy as np
import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
from message_filters import ApproximateTimeSynchronizer, Subscriber

image_buffer=[]
MAX_BUFFER_SIZE=10





class ImageNode(Node):
    def __init__(self):
        super().__init__('NODE')
        print("node")
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.callback,
            10
        )
        # self.time_sub = Subscriber(self, Clock, '/clock')
        # self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.time_sub], queue_size=10, slop=0.1)
        print("test")


        self.img_size = 640
        self.stride = 32
        self.auto = True


    def callback(self, rgb_msg):
        print("$$$$$$$$$$$$$$in")
        timestamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9


        rgb_image = self.ros_img_to_numpy(rgb_msg)


        # im = np.stack([self.letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in rgb_image])  # resize
        # im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        # im = np.ascontiguousarray(im)  # contiguous



        if len(self.image_buffer) >= MAX_BUFFER_SIZE:
            image_buffer.pop(0)

        image_buffer.append((rgb_image, timestamp))


    def ros_img_to_numpy(self, img_msg):
        # Convert ROS Image message to NumPy array
        np_arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
        return np_arr
