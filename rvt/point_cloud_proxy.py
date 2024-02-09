from typing import Tuple, Optional
from dataclasses import dataclass
from numpy.typing import NDArray
import functools

from threading import Lock
import numpy as np
import matplotlib.pyplot as plt
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2
from skimage.transform import resize
import tf as not_tensorflow
import tf2_ros
import utils

from online_isec.tf_proxy import TFProxy


@dataclass
class PointCloudProxy:
    pc_topics: Tuple[str, ...] = ("/camera_left/depth/color/points", "/camera_right/depth/color/points")
    heights: Tuple[int, ...] = (720, 720)
    widths: Tuple[int, ...] = (1280, 1280)
    nans_in_pc: Tuple[bool, ...] = (False, False)

    def __post_init__(self):

        self.cloud = None

        self.tf_proxy = TFProxy()

        self.msgs = [None for _ in range(len(self.pc_topics))]
        # XYZRGB point cloud.
        self.clouds = [None for _ in range(len(self.pc_topics))]
        # Flatten image and apply mask to turn it into a point cloud.
        # Useful if we want to have correspondence between a predicted segmentation mask
        # and the point cloud.
        self.masks = [None for _ in range(len(self.pc_topics))]
        self.locks = [Lock() for _ in range(len(self.pc_topics))]

        self.pc_subs = []
        for i in range(len(self.pc_topics)):
            save_image_from_pc = (self.image_topics[i] is None) and self.save_image[i]
            # Get point clouds.
            self.pc_subs.append(rospy.Subscriber(self.pc_topics[i], PointCloud2, functools.partial(
                self.pc_callback, camera_index=i, width=self.widths[i], height=self.heights[i], nans_in_pc=self.nans_in_pc[i]
            ), queue_size=1))

    def pc_callback(self, msg: rospy.Message, camera_index: int, width: int, height: int, nans_in_pc: bool):

        # Get XYZRGB point cloud from a message.
        cloud_frame = msg.header.frame_id
        pc = ros_numpy.numpify(msg)
        if save_image:
            pc = ros_numpy.point_cloud2.split_rgb_field(pc)

        # Get point cloud.
        cloud = np.zeros((height * width, 6), dtype=np.float32)
        cloud[:, 0] = np.resize(pc["x"], height * width)
        cloud[:, 1] = np.resize(pc["y"], height * width)
        cloud[:, 2] = np.resize(pc["z"], height * width)

        # Get point colors.
        cloud[:, 3] = np.resize(pc["r"], height * width)
        cloud[:, 4] = np.resize(pc["g"], height * width)
        cloud[:, 5] = np.resize(pc["b"], height * width)

        # We'll keep point cloud RGB values in 0-1 floats.
        cloud[:, 3: 6] /= 255.

        if nans_in_pc:
            # Mask out NANs and keep the mask so that we can go from image to PC.
            # TODO: I added ..., 3 here, double check if there are NaNs in colors.
            mask = np.logical_not(np.isnan(cloud[..., :3]).any(axis=1))
        else:
            # If empty pixels are not NaN they should be (0, 0, 0).
            # Note the corresponding RGB values will not be NaN.
            mask = np.logical_not((cloud[..., :3] == 0).all(axis=1))

        cloud = cloud[mask]

        T = self.tf_proxy.lookup_transform(cloud_frame, "panda_link0", rospy.Time(0))
        cloud[:, :3] = utils.transform_pointcloud_2(cloud[:, :3], T)

        with self.locks[camera_index]:
            self.msgs[camera_index] = msg
            self.clouds[camera_index] = cloud
            if save_image:
                self.images[camera_index] = image
            self.masks[camera_index] = mask

    def get(self, camera_index: int) -> Tuple[Optional[NDArray], Optional[NDArray], Optional[NDArray]]:

        with self.locks[camera_index]:
            return self.clouds[camera_index], self.masks[camera_index]
