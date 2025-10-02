import numpy as np
from typing import List
from scipy.interpolate import interp1d
import carla
from .bev import BEV

class PurePursuit():
    def __init__(self, camera_spawnpoint, wheelbase, rear_axle_offset, ego_vehicle):
        self.bev = BEV(camera_spawnpoint)
        self.wheelbase = wheelbase
        self.rear_axle_offset = rear_axle_offset
        self.ego_vehicle = ego_vehicle

        self.k_v = 0.4
        self.ld_min = 3.0
        self.ld_max = 20.0

    
    def get_centerline(self, left_lanemarking: np.ndarray, right_lanemarking: np.ndarray, lane_width):
        """
        lanemarkings are a list of x, y coordinates in rear axle vehicle frame
        """
        if left_lanemarking.size != 0 and right_lanemarking.size != 0:
            # Sort both markings by x to ensure interp1d works properly
            left_sorted = left_lanemarking[np.argsort(left_lanemarking[:, 0])]
            right_sorted = right_lanemarking[np.argsort(right_lanemarking[:, 0])]

            # Determine common x-range for interpolation
            min_x = max(left_sorted[0, 0], right_sorted[0, 0])
            max_x = min(left_sorted[-1, 0], right_sorted[-1, 0])
            common_x = np.linspace(min_x, max_x, 80)

            # Interpolate y for both lanes at common_x
            left_interp = interp1d(left_sorted[:, 0], left_sorted[:, 1], kind='linear', fill_value='extrapolate', assume_sorted=True)
            right_interp = interp1d(right_sorted[:, 0], right_sorted[:, 1], kind='linear', fill_value='extrapolate', assume_sorted=True)

            left_y = left_interp(common_x)
            right_y = right_interp(common_x)

            # Compute centerline y by averaging left and right y
            centerline_y = (left_y + right_y) / 2

            return np.stack([common_x, centerline_y], axis=1) # (N, 2)
        else:
            return np.array([])
        
    
    def get_lookahead_distance(self, speed):
        """
        speed in m/s in vehicle frame
        """
        ld = self.k_v * speed
        return np.clip(ld, self.ld_min, self.ld_max)
    

    def find_target_point(self, path_points: np.ndarray, lookahead_distance):
        """
        path_points should be in rear axle vehicle frame
        """
        # Distance from rear axle (0,0)
        dists = np.linalg.norm(path_points, axis=1)

        # Find closest point beyond the lookahead
        candidates = path_points[dists > lookahead_distance]
        if len(candidates) > 0:
            target = candidates[0]
        else:
            target = path_points[-1]  # fallback

        return target
    

    def compute_steering_angle(self, target):
        # Pure Pursuit formula
        # Assume rear axle at (0, 0)
        x = target[0]
        y = target[1]
        ld = np.sqrt(x**2 + y**2)
        steering_angle = np.arctan2(2.0 * self.wheelbase * y, ld**2)
        return steering_angle
    

    def run(self, lanes_list_processed, image_depth, speed):
        lanes_rear = self.bev.pixel_to_rear(lanes_list_processed, image_depth, self.rear_axle_offset)
        centerline = self.get_centerline(lanes_rear[1], lanes_rear[2], None) # ego lanes
        lanes_rear.insert(2, centerline)

        if len(centerline) > 0:
            # compute lookahead distance
            ld = self.get_lookahead_distance(speed)

            # find target point
            target = self.find_target_point(centerline, ld)
            self.bev.get_bev_view(lanes_rear, target)

            # compute steering angle
            steering_angle = self.compute_steering_angle(target)
            return np.clip(steering_angle, -1.0, 1.0)

        else:
            print("NOT CONTROLLING")
            return 0


