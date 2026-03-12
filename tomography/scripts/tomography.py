#!/usr/bin/python3
import os
import sys
import time
import pickle
import numpy as np
import open3d as o3d
  
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from tomogram import Tomogram

sys.path.append('../')
from config import POINT_FIELDS_XYZI, GRID_POINTS_XYZI
from config import Config

rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'


class Tomography(object):
    def __init__(self, cfg, scene_cfg):
        self.export_dir = rsg_root + cfg.map.export_dir
        self.pcd_file = scene_cfg.pcd.file_name
        self.auto_align_ground = getattr(scene_cfg.pcd, 'auto_align_ground', False)
        self.ground_seed_percentile = float(getattr(scene_cfg.pcd, 'ground_seed_percentile', 35.0))
        self.ground_ransac_dist = float(getattr(scene_cfg.pcd, 'ground_ransac_dist', 0.08))
        self.ground_ransac_n = int(getattr(scene_cfg.pcd, 'ground_ransac_n', 3))
        self.ground_ransac_iters = int(getattr(scene_cfg.pcd, 'ground_ransac_iters', 1000))
        self.pcd_rot_deg = np.asarray(getattr(scene_cfg.pcd, 'rot_deg', [0.0, 0.0, 0.0]), dtype=np.float32)
        self.resolution = scene_cfg.map.resolution
        self.ground_h = scene_cfg.map.ground_h
        self.slice_dh = scene_cfg.map.slice_dh

        self.center = np.zeros(2, dtype=np.float32)
        self.tomogram = Tomogram(scene_cfg)
        points = self.loadPCD(self.pcd_file)

        # Process
        self.process(points)

    def initROS(self):
        # cfg.ros.map_frame='map'
        # self.map_frame = cfg.ros.map_frame
        self.map_frame = cfg.ros.map_frame

        pointcloud_topic = cfg.ros.pointcloud_topic
        self.pointcloud_pub = rospy.Publisher(pointcloud_topic, PointCloud2, latch=True, queue_size=1)

        self.layer_G_pub_list = []
        self.layer_C_pub_list = []
        layer_G_topic = cfg.ros.layer_G_topic
        layer_C_topic = cfg.ros.layer_C_topic
        for i in range(self.n_slice):
            layer_G_pub = rospy.Publisher(layer_G_topic + str(i), PointCloud2, latch=True, queue_size=1)
            self.layer_G_pub_list.append(layer_G_pub)
            layer_C_pub = rospy.Publisher(layer_C_topic + str(i), PointCloud2, latch=True, queue_size=1)
            self.layer_C_pub_list.append(layer_C_pub)

        tomogram_topic = cfg.ros.tomogram_topic
        self.tomogram_pub = rospy.Publisher(tomogram_topic, PointCloud2, latch=True, queue_size=1)

    def loadPCD(self, pcd_file):
        pcd = o3d.io.read_point_cloud(rsg_root + "/rsc/pcd/" + pcd_file)
        points = np.asarray(pcd.points).astype(np.float32)
        rospy.loginfo("PCD points: %d", points.shape[0])

        if points.shape[1] > 3:
            points = points[:, :3]

        points = self.alignGroundPlane(points)
        points = self.correctPointCloudTilt(points)
        self.points_max = np.max(points, axis=0)
        self.points_min = np.min(points, axis=0)           
        self.points_min[-1] = self.ground_h
        self.map_dim_x = int(np.ceil((self.points_max[0] - self.points_min[0]) / self.resolution)) + 4
        self.map_dim_y = int(np.ceil((self.points_max[1] - self.points_min[1]) / self.resolution)) + 4
        n_slice_init = int(np.ceil((self.points_max[2] - self.points_min[2]) / self.slice_dh))
        self.center = (self.points_max[:2] + self.points_min[:2]) / 2
        self.slice_h0 = self.points_min[-1] + self.slice_dh
        self.tomogram.initMappingEnv(self.center, self.map_dim_x, self.map_dim_y, n_slice_init, self.slice_h0)

        rospy.loginfo("Map center: [%.2f, %.2f]", self.center[0], self.center[1])
        rospy.loginfo("Dim_x: %d", self.map_dim_x)
        rospy.loginfo("Dim_y: %d", self.map_dim_y)
        rospy.loginfo("Num slices init: %d", n_slice_init)

        self.VISPROTO_I, self.VISPROTO_P = \
            GRID_POINTS_XYZI(self.resolution, self.map_dim_x, self.map_dim_y)

        return points

    def alignGroundPlane(self, points):
        if not self.auto_align_ground:
            return points

        z_threshold = np.percentile(points[:, 2], self.ground_seed_percentile)
        seed_points = points[points[:, 2] <= z_threshold]
        if seed_points.shape[0] < self.ground_ransac_n:
            rospy.logwarn(
                "Skip ground auto-alignment: not enough seed points (%d)",
                seed_points.shape[0]
            )
            return points

        seed_pcd = o3d.geometry.PointCloud()
        seed_pcd.points = o3d.utility.Vector3dVector(seed_points.astype(np.float64))
        plane_model, inliers = seed_pcd.segment_plane(
            distance_threshold=self.ground_ransac_dist,
            ransac_n=self.ground_ransac_n,
            num_iterations=self.ground_ransac_iters,
        )

        normal = np.asarray(plane_model[:3], dtype=np.float32)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            rospy.logwarn("Skip ground auto-alignment: invalid plane normal")
            return points

        normal /= normal_norm
        if normal[2] < 0.0:
            normal = -normal

        target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        rot = self.rotationMatrixFromVectors(normal, target)
        center = points.mean(axis=0, keepdims=True)
        aligned = (points - center) @ rot.T + center

        tilt_deg = np.rad2deg(np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0)))
        rospy.loginfo(
            "Ground auto-alignment: normal=[%.4f, %.4f, %.4f], tilt=%.3f deg, seed_points=%d, inliers=%d",
            normal[0], normal[1], normal[2], tilt_deg, seed_points.shape[0], len(inliers)
        )
        return aligned

    def correctPointCloudTilt(self, points):
        if np.allclose(self.pcd_rot_deg, 0.0):
            return points

        roll, pitch, yaw = np.deg2rad(self.pcd_rot_deg)
        cx, sx = np.cos(roll), np.sin(roll)
        cy, sy = np.cos(pitch), np.sin(pitch)
        cz, sz = np.cos(yaw), np.sin(yaw)

        rot_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ], dtype=np.float32)
        rot_y = np.array([
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ], dtype=np.float32)
        rot_z = np.array([
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        rot = rot_z @ rot_y @ rot_x
        center = points.mean(axis=0, keepdims=True)
        corrected = (points - center) @ rot.T + center

        rospy.loginfo(
            "Apply point cloud rotation correction [roll_x, pitch_y, yaw_z] = [%.3f, %.3f, %.3f] deg",
            self.pcd_rot_deg[0], self.pcd_rot_deg[1], self.pcd_rot_deg[2]
        )
        return corrected

    def rotationMatrixFromVectors(self, src, dst):
        src = src / np.linalg.norm(src)
        dst = dst / np.linalg.norm(dst)
        cross = np.cross(src, dst)
        cross_norm = np.linalg.norm(cross)
        dot = np.clip(np.dot(src, dst), -1.0, 1.0)

        if cross_norm < 1e-6:
            if dot > 0.0:
                return np.eye(3, dtype=np.float32)

            axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(src[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            axis = axis - src * np.dot(axis, src)
            axis = axis / np.linalg.norm(axis)
            return self.axisAngleToMatrix(axis, np.pi)

        axis = cross / cross_norm
        angle = np.arccos(dot)
        return self.axisAngleToMatrix(axis, angle)

    def axisAngleToMatrix(self, axis, angle):
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        one_c = 1.0 - c

        return np.array([
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ], dtype=np.float32)
        
    def process(self, points):        
        t_map = 0.0
        t_trav = 0.0
        t_simp = 0.0
        t_all = 0.0
        n_repeat = 10

        """ 
        GPU time benchmark, where CUDA events are synchronized for correct time measurement.
        The function is repeatedly run for n_repeat times to calculate the average processing time of each modules.
        The time of the first warm-up run is excluded to reduce timing fluctuation and exclude the overhead in initial invocations.
        See https://docs.cupy.dev/en/stable/user_guide/performance.html for more details
        """
        for i in range(n_repeat + 1):
            t_start = time.time()
            layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c, t_gpu = self.tomogram.point2map(points)

            if i > 0:
                t_map += t_gpu['t_map']
                t_trav += t_gpu['t_trav']
                t_simp += t_gpu['t_simp']
                t_all += (time.time() - t_start) * 1e3

        rospy.loginfo("Num slices simp: %d", layers_g.shape[0])
        rospy.loginfo("Num repeats (for benchmarking only): %d", n_repeat)
        rospy.loginfo(" -- avg t_map  (ms): %f", t_map / n_repeat)
        rospy.loginfo(" -- avg t_trav (ms): %f", t_trav / n_repeat)
        rospy.loginfo(" -- avg t_simp (ms): %f", t_simp / n_repeat)
        rospy.loginfo(" -- avg t_all  (ms): %f", t_all / n_repeat)

        self.n_slice = layers_g.shape[0]

        map_file = os.path.splitext(self.pcd_file)[0]
        self.exportTomogram(np.stack((layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c)), map_file)

        self.initROS()
        self.publishPoints(points)
        self.publishLayers(self.layer_G_pub_list, layers_g, layers_t)
        self.publishLayers(self.layer_C_pub_list, layers_c, None)
        self.publishTomogram(layers_g, layers_t)

    def exportTomogram(self, tomogram, map_file):        
        data_dict = {
            'data': tomogram.astype(np.float16),
            'resolution': self.resolution,
            'center': self.center,
            'slice_h0': self.slice_h0,
            'slice_dh': self.slice_dh,
        }
        file_name = map_file + '.pickle'
        with open(self.export_dir + file_name, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        rospy.loginfo("Tomogram exported: %s", file_name)

    def publishPoints(self, points):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.map_frame

        point_msg = pc2.create_cloud_xyz32(header, points)
        self.pointcloud_pub.publish(point_msg)

    def publishLayers(self, pub_list, layers, color=None):
        header = Header()
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = self.map_frame

        layer_points = self.VISPROTO_P.copy()
        layer_points[:, :2] += self.center

        for i in range(layers.shape[0]):
            layer_points[:, 2] = layers[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            if color is not None:
                layer_points[:, 3] = color[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            else:
                layer_points[:, 3] = 1.0
        
            valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
            points_msg = pc2.create_cloud(header, POINT_FIELDS_XYZI, valid_points)
            pub_list[i].publish(points_msg) 

    def publishTomogram(self, layers_g, layers_t):
        header = Header()
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = self.map_frame

        n_slice = layers_g.shape[0]
        vis_g = layers_g.copy()
        vis_t = layers_t.copy() 
        layer_points = self.VISPROTO_P.copy()
        layer_points[:, :2] += self.center

        global_points = None
        for i in range(n_slice - 1):
            mask_h = (vis_g[i + 1] - vis_g[i]) < self.slice_dh
            vis_g[i, mask_h] = np.nan
            vis_t[i + 1, mask_h] = np.minimum(vis_t[i, mask_h], vis_t[i + 1, mask_h])
            layer_points[:, 2] = vis_g[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            layer_points[:, 3] = vis_t[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
            if global_points is None:
                global_points = valid_points
            else:
                global_points = np.concatenate((global_points, valid_points), axis=0)

        layer_points[:, 2] = vis_g[-1, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
        layer_points[:, 3] = vis_t[-1, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
        valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
        global_points = np.concatenate((global_points, valid_points), axis=0)
        
        points_msg = pc2.create_cloud(header, POINT_FIELDS_XYZI, global_points)
        self.tomogram_pub.publish(points_msg)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, help='Name of the scene. Available: [\'Spiral\', \'Building\', \'Plaza\']')
    args = parser.parse_args()

    cfg = Config()
    scene_cfg = getattr(__import__('config'), 'Scene' + args.scene)

    rospy.init_node('pointcloud_tomography', anonymous=True)

    print(scene_cfg.pcd.file_name)

    mapping = Tomography(cfg, scene_cfg)

    rospy.spin()
