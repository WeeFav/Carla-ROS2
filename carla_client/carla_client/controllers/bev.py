import carla
import numpy as np   
import config as cfg 
import math
import cv2
from typing import List

class BEV():
    def __init__(self, camera_spawnpoint):
        self.c_x = cfg.image_width / 2
        self.c_y = cfg.image_height / 2
        self.f = cfg.image_width / (2 * math.tan(cfg.fov * math.pi / 360))

        self.vehicle_to_camera_matrix = self.get_vehicle_to_camera_matrix(camera_spawnpoint)
        self.camera_to_vehicle_matrix = np.linalg.inv(self.vehicle_to_camera_matrix)

        # BEV area in meters (vehicle-centered)
        self.x_range = (-10, 40)  # From 10 meters behind to 40 meters front
        self.y_range = (-20, 20)  # From 20 meters left to 20 meters right
        self.resolution = 0.1     # meters per pixel (i.e., 10 pixels/m)

        # Image size
        self.bev_width  = int((self.y_range[1] - self.y_range[0]) / self.resolution)   # columns
        self.bev_height = int((self.x_range[1] - self.x_range[0]) / self.resolution)   # rows

        self.BGR_colors = [(0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 255, 255), (255, 0, 0)]


    def get_vehicle_to_camera_matrix(self, camera_spawnpoint):
        rotation = camera_spawnpoint.rotation
        location = camera_spawnpoint.location

        # Convert CARLA rotation (in degrees) to radians and then to matrix
        pitch = math.radians(rotation.pitch)
        yaw   = math.radians(rotation.yaw)
        roll  = math.radians(rotation.roll)

        # Create rotation matrix from pitch, yaw, roll
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])

        R_yaw = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        R_roll = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])

        R_rotation = R_roll @ R_yaw @ R_pitch

        # Rotation matrix for axes conversion (vehicle to camera)
        R_axes = [
            [0, 1, 0],  # z_camera = x_vehicle
            [0, 0, -1], # x_camera = y_vehicle
            [1, 0, 0]   # y_camera = -z_vehicle
        ]

        R = R_rotation @ R_axes

        # T_vehicle is the translation of the camera in vehicle frame
        # However, the vehicle_to_camera_matrix do rotation first then translation, which means a point is rotated into camera frame first, then translate according to camera axes
        # So the translation must be in camera frame 
        T_vehicle = np.array([location.x, location.y, location.z]).reshape((3, 1))
        T_camera = R @ T_vehicle

        # Combine into 4x4 matrix
        vehicle_to_camera_matrix = np.eye(4)
        vehicle_to_camera_matrix[:3, :3] = R
        vehicle_to_camera_matrix[:3, 3] = T_camera.flatten()

        return vehicle_to_camera_matrix


    def pixel_to_camera(self, points_2d, image_depth):
        x_coords = points_2d[:, 0].astype(int) # (N,)
        y_coords = points_2d[:, 1].astype(int)

        rgb = image_depth[y_coords, x_coords] # (N, 3)
        R = rgb[:, 0].astype(int) # (N,)
        G = rgb[:, 1].astype(int)
        B = rgb[:, 2].astype(int)

        normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        depth_in_meters = 1000 * normalized

        x_camera = (x_coords - self.c_x) * depth_in_meters / self.f
        y_camera = (y_coords - self.c_y) * depth_in_meters / self.f
        z_camera = depth_in_meters

        points_camera = np.stack([x_camera, y_camera, z_camera], axis=1)

        return points_camera


    def camera_to_vehicle(self, points_camera):
        homo = np.ones((points_camera.shape[0], 1))
        points_camera = np.concatenate((points_camera, homo), axis=1)
        points_vehicle = (self.camera_to_vehicle_matrix @ points_camera.T).T
        return points_vehicle[:, :3] # (N, 3)
    

    def vehicle_to_rear(self, points_vehicle, rear_axle_offset):
        points_vehicle[:, 0] = points_vehicle[:, 0] - rear_axle_offset
        return points_vehicle
    

    def pixel_to_rear(self, lanes_list_processed, image_depth, rear_axle_offset) -> List[np.ndarray]:
        lanes_rear = []

        for i in range(len(lanes_list_processed)):
            if lanes_list_processed[i]:
                points_2d = np.array(lanes_list_processed[i])
                points_camera = self.pixel_to_camera(points_2d, image_depth)
                points_vehicle = self.camera_to_vehicle(points_camera) # (N, 3)
                points_rear = self.vehicle_to_rear(points_vehicle, rear_axle_offset)
                xs = points_rear[:, 0]
                ys = points_rear[:, 1]
                pts = np.stack([xs, ys], axis=1) # (N, 2)
            else:
                pts = np.array([])
            lanes_rear.append(pts)
        
        return lanes_rear


    def rear_to_bev(self, xs, ys):
        us = ((ys - self.y_range[0]) / self.resolution).astype(np.int32)  # left to right becomes u=0 to width
        vs = ((self.x_range[1] - xs) / self.resolution).astype(np.int32)  # front to back becomes v=0 to height
        return us, vs


    def get_bev_view(self, lanes_rear: List[np.ndarray], target):
        """
        lanes_rear should have 5 lanes: outer left, left, centerline, right, outer right
        """
        bev_image = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)

        # rear axle origin
        u0, v0 = self.rear_to_bev(np.array([0]), np.array([0]))
        cv2.circle(bev_image, center=(u0[0], v0[0]), radius=3, color=(0, 0, 255), thickness=-1)  

        # target
        ut, vt = self.rear_to_bev(np.array([target[0]]), np.array([target[1]]))
        cv2.circle(bev_image, center=(ut[0], vt[0]), radius=3, color=(0, 255, 0), thickness=-1)     

        for i in range(len(lanes_rear)):
            points_rear = lanes_rear[i]
            if points_rear.size != 0:
                xs = points_rear[:, 0]
                ys = points_rear[:, 1]

                mask = (
                    (xs >= self.x_range[0]) & (xs <= self.x_range[1]) &
                    (ys >= self.y_range[0]) & (ys <= self.y_range[1])
                )

                xs_valid = xs[mask]
                ys_valid = ys[mask]

                us, vs = self.rear_to_bev(xs_valid, ys_valid)
                pts = np.stack([us, vs], axis=1)               # shape (N, 2)
                pts = pts.reshape((-1, 1, 2)).astype(np.int32) # shape (N, 1, 2), int32 type

                cv2.polylines(bev_image, [pts], isClosed=False, color=self.BGR_colors[i], thickness=1)                

        cv2.imshow("BEV", bev_image)
        cv2.waitKey(1)