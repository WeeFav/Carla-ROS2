import numpy as np
import math

def get_corners(vehicle, world_to_lidar_mat):
    bb = vehicle.bounding_box

    center_vehicle = np.array([bb.location.x, bb.location.y, bb.location.z])
    
    # --- Define the 8 corners in vehicle local frame ---
    # Order convention:
    # [0-3]: bottom rectangle (front-left, front-right, rear-right, rear-left)
    # [4-7]: top rectangle (front-left, front-right, rear-right, rear-left)
    x = bb.extent.x
    y = bb.extent.y
    z = bb.extent.z

    corners_local = np.array([
        [ x, -y, -z],   # front-left-bottom
        [ x,  y, -z],   # front-right-bottom
        [-x,  y, -z],   # rear-right-bottom
        [-x, -y, -z],   # rear-left-bottom
        [ x, -y,  z],   # front-left-top
        [ x,  y,  z],   # front-right-top
        [-x,  y,  z],   # rear-right-top
        [-x, -y,  z],   # rear-left-top
    ])

    # Apply bounding box center offset
    corners_local += center_vehicle

    # Homogeneous coordinates
    corners_local_h = np.hstack((corners_local, np.ones((8, 1))))

    # vehicle to world
    vehicle_to_world_mat = np.array(vehicle.get_transform().get_matrix())
    corners_world = (vehicle_to_world_mat @ corners_local_h.T).T

    # world to lidar
    corners_lidar = (world_to_lidar_mat @ corners_world.T).T[:, :3]

    return corners_lidar


def is_visible_by_lidar(corners_lidar, pointcloud, min_points=10):
    p0, p1, p3 = corners_lidar[0], corners_lidar[1], corners_lidar[3]
    
    # Axes of the box
    x_axis = (p1 - p0) / np.linalg.norm(p1 - p0)      # length direction
    y_axis = (p3 - p0) / np.linalg.norm(p3 - p0)      # width direction
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    
    R = np.vstack([x_axis, y_axis, z_axis]).T
    origin = p0
    
    # Transform points into box coordinates
    pts_local = (pointcloud - origin) @ R
    
    # Box dimensions
    length = np.linalg.norm(p1 - p0)
    width  = np.linalg.norm(p3 - p0)
    height = np.linalg.norm(corners_lidar[4] - corners_lidar[0])
    
    # Check if inside
    mask = (
        (pts_local[:, 0] >= 0) & (pts_local[:, 0] <= length) &
        (pts_local[:, 1] >= 0) & (pts_local[:, 1] <= width) &
        (pts_local[:, 2] >= 0) & (pts_local[:, 2] <= height)
    )
    
    return np.sum(mask) >= min_points


def get_bboxes(world, pointcloud, sensor_lidar):
    """
    Given pointcloud, return a list of bounding box annotations that are visible in the pointcloud.
                
    Returns a list of dictionary:
        corners_lidar: (8, 3) np.ndarray, ordered corners in LiDAR frame
        bottom_center: (3,) np.ndarray, xyz coordinate of the bottom center of the object in lidar frame
        dims: (3,) np.ndarray, height, width, length in object frame
        rotation_z: rotation around the lidar frame's z axis in radians

    To get 8 corners in lidar frame, first find 8 corners with height, width, length, then rotate around z-axis, then translate to center
    """

    world_to_lidar_mat = np.array(sensor_lidar.transform.get_inverse_matrix())

    lidar_bboxes = []
    labels = []
    
    for vehicle in world.get_actors().filter('*vehicle*'):
        # Skip ego vehicle
        role_name = vehicle.attributes.get("role_name", "")
        if role_name == "hero":
            continue

        # Skip vehicle with distance > 100 m
        bb = vehicle.bounding_box
        center_vehicle = np.array([bb.location.x, bb.location.y, bb.location.z, 1.0])
        vehicle_to_world_mat = np.array(vehicle.get_transform().get_matrix())
        center_world = vehicle_to_world_mat @ center_vehicle
        center_lidar = (world_to_lidar_mat @ center_world)[:3]
        if np.linalg.norm(center_lidar) > 100:
            continue
        
        corners_lidar = get_corners(vehicle, world_to_lidar_mat)

        if is_visible_by_lidar(corners_lidar, pointcloud):
            length = bb.extent.x * 2
            width = bb.extent.y * 2
            height = bb.extent.z * 2

            bottom_center = corners_lidar[0:4].mean(axis=0)

            # Compute yaw in lidar frame
            # Compute midpoints of front and rear edges
            front_mid = (corners_lidar[0] + corners_lidar[1]) / 2.0
            rear_mid  = (corners_lidar[2] + corners_lidar[3]) / 2.0

            # Forward vector points from rear to front
            forward_vec = front_mid - rear_mid
            yaw = np.arctan2(forward_vec[1], forward_vec[0])

            if vehicle.attributes['base_type'] == 'bicycle':
                object_type = 'Cyclist'
            else:
                object_type = 'Car'

            lidar_bboxes.append(np.concatanate([bottom_center, np.array([height, width, length, yaw])], axis=0))
            labels.append(object_type)
    
    return np.array(lidar_bboxes), np.array(labels) # (N, 7), (N,)