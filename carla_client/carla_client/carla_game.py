#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import carla
import random
import pygame
import time
import numpy as np
import cv2
import message_filters
import threading
import open3d as o3d
import traceback

from sensor_msgs.msg import Image, PointCloud2   
from carla_client_msgs.msg import Lanes

from carla_client.vehicle_manager import VehicleManager
from carla_client.lanemarkings import LaneMarkings
from carla_client.lane_detection.lanedet import LaneDet
from carla_client.lidar_object_detection import get_bboxes, bbox3d2corners

class CarlaGame(Node):
    def __init__(self):
        super().__init__('carla_game')

        # Get parameters
        self.host = self.declare_parameter("host", "0.0.0.0").value
        self.town = self.declare_parameter("town", "Town10HD_Opt").value
        self.num_vehicles = self.declare_parameter("num_vehicles", 50).value
        self.respawn = self.declare_parameter("respawn", 50).value
        self.carla_auto_pilot = self.declare_parameter("carla_auto_pilot", True).value
        self.predict_lane = self.declare_parameter("predict_lane", False).value
        self.predict_object = self.declare_parameter("predict_object", False).value
        self.compute_lanemarkings = self.declare_parameter("compute_lanemarkings", False).value
        self.compute_bbox = self.declare_parameter("compute_bbox", False).value
        self.enable_pid = self.declare_parameter("enable_pid", False).value
        self.enable_pure_pursuit = self.declare_parameter("enable_pure_pursuit", False).value

        # Fixed parameters
        self.weather = carla.WeatherParameters.ClearNoon
        self.fps = 10
        self.image_width = 1280
        self.image_height = 720
        self.fov = 90
        self.number_of_lanepoints = 80
        self.meters_per_frame = 1.0
        self.render_lanes = True

        # Set up pygame
        self.display = pygame.display.set_mode((self.image_width, self.image_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pygame_clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 36) 

        # Connect client
        self.client = carla.Client(self.host, 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(self.town)
        self.world.set_weather(self.weather)
        self.map = self.world.get_map()
        
        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 10
        self.world.apply_settings(settings)

        # Vehicle manager
        self.tm = self.client.get_trafficmanager()
        self.tm.set_synchronous_mode(True)
        self.vehicle_manager = VehicleManager(self.world, self.tm, self.fps, self.respawn, self.num_vehicles)

        # Ego vehicle setup
        self.ego_vehicle = self.vehicle_manager.spawn_ego_vehicle(self.carla_auto_pilot)

        blueprint_library = self.world.get_blueprint_library()
        self.sensors = []

        # Spawn rgb-cam and attach to vehicle
        bp_camera_rgb = blueprint_library.find('sensor.camera.rgb')
        bp_camera_rgb.set_attribute('image_size_x', f'{self.image_width}')
        bp_camera_rgb.set_attribute('image_size_y', f'{self.image_height}')
        bp_camera_rgb.set_attribute('fov', f'{self.fov}')
        bp_camera_rgb.set_attribute('role_name', 'rgb')
        self.camera_spawnpoint = carla.Transform(carla.Location(x=1.0, z=2.0), carla.Rotation(pitch=-18.5)) # camera 5
        self.camera_rgb = self.world.spawn_actor(bp_camera_rgb, self.camera_spawnpoint, attach_to=self.ego_vehicle)
        self.sensors.append(self.camera_rgb)

        if self.compute_lanemarkings:
            # Spawn semseg-cam and attach to vehicle
            bp_camera_semseg = blueprint_library.find('sensor.camera.semantic_segmentation')
            bp_camera_semseg.set_attribute('image_size_x', f'{self.image_width}')
            bp_camera_semseg.set_attribute('image_size_y', f'{self.image_height}')
            bp_camera_semseg.set_attribute('fov', f'{self.fov}')
            bp_camera_semseg.set_attribute('role_name', 'semseg')
            self.camera_semseg = self.world.spawn_actor(bp_camera_semseg, self.camera_spawnpoint, attach_to=self.ego_vehicle)
            self.sensors.append(self.camera_semseg)

        if self.enable_pure_pursuit:
            # Spawn depth-cam and attach to vehicle
            bp_camera_depth = blueprint_library.find('sensor.camera.depth')
            bp_camera_depth.set_attribute('image_size_x', f'{self.image_width}')
            bp_camera_depth.set_attribute('image_size_y', f'{self.image_height}')
            bp_camera_depth.set_attribute('fov', f'{self.fov}')
            bp_camera_depth.set_attribute('role_name', 'depth')
            self.camera_depth = self.world.spawn_actor(bp_camera_depth, self.camera_spawnpoint, attach_to=self.ego_vehicle)
            self.sensors.append(self.camera_depth)

        # Spawn lidar and attach to vehicle
        bp_lidar = blueprint_library.find("sensor.lidar.ray_cast")
        bp_lidar.set_attribute("range", "120") # 120 meter range for cars and foliage
        bp_lidar.set_attribute("rotation_frequency", "10")
        bp_lidar.set_attribute("channels", "64") # vertical resolution of the laser scanner is 64
        bp_lidar.set_attribute("points_per_second", "1300000")
        bp_lidar.set_attribute("upper_fov", "2.0") # +2 up to -24.8 down
        bp_lidar.set_attribute("lower_fov", "-24.8")
        bp_lidar.set_attribute('role_name', 'lidar')
        self.lidar_spawnpoint = carla.Transform(carla.Location(x=0, y=0, z=1.73))
        self.lidar = self.world.spawn_actor(bp_lidar, self.lidar_spawnpoint, attach_to=self.ego_vehicle)
        self.sensors.append(self.lidar)

        # Spawn other vehicles
        self.vehicle_manager.spawn_vehicles()

        self.tick_counter = 0

        self.RGB_colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 0, 255)]
        self.BGR_colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 0)]

        # Create opencv window
        cv2.namedWindow("inst_background", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("inst_background", 640, 360)

        # Initialize Open3D Visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='CARLA LiDAR', width=800, height=600)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(np.random.rand(10, 3))
        self.vis.add_geometry(self.pcd)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
        self.vis.add_geometry(mesh_frame)

        render_opt = self.vis.get_render_option()
        render_opt.background_color = np.asarray([0, 0, 0])
        render_opt.point_size = 1
        
        ctr = self.vis.get_view_control()
        ctr.change_field_of_view(step=90)
        ctr.set_constant_z_far(2000)
        ctr.set_constant_z_near(0.1)
        self.vis.reset_view_point(True)
        self.cam = ctr.convert_to_pinhole_camera_parameters()

        self.bbox_lines = []

        # Define line connections between the 8 corners
        self.lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom rectangle
            [4, 5], [5, 6], [6, 7], [7, 4],  # top rectangle
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]


        if self.predict_lane:
            self.lanedet = LaneDet()

        if self.compute_lanemarkings:
            self.lanemarkings = LaneMarkings(self.world, self.image_width, self.image_height, self.fov, self.number_of_lanepoints, self.meters_per_frame)


        # Subscribers
        self.camera_rgb_sub = message_filters.Subscriber(self, Image, '/carla/hero/rgb/image')
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, '/carla/hero/lidar')
        self.synchronizer_subs = [self.camera_rgb_sub, self.lidar_sub]

        if self.compute_lanemarkings:
            self.camera_semseg_sub = message_filters.Subscriber(self, Image, '/carla/hero/semseg/image')
            self.synchronizer_subs.append(self.camera_semseg_sub)
        if self.enable_pure_pursuit:
            self.camera_depth_sub = message_filters.Subscriber(self, Image, '/carla/hero/depth/image')
            self.synchronizer_subs.append(self.camera_depth_sub)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.synchronizer_subs,
            queue_size=10,
            slop=0.05   # allowed timestamp difference in seconds
        )
        self.ts.registerCallback(self.sync_callback)

        self.sync_done = threading.Event()

        self.modules_event = []

        if self.predict_lane:
            self.pred_sub = self.create_subscription(Lanes, '/lanes', self.lanes_prediction_callback, 10)
            self.lane_prediction_done = threading.Event()
            self.modules_event.append(self.lane_prediction_done)


    def reshape_image(self, image):
        array = np.frombuffer(image.data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4)) # BGRA
        array = array[:, :, :3] # BGR
        array = array[:, :, ::-1] # RGB, (H, W, C)
        return array # (H, W, C)
    

    def reshape_pointcloud(self, pointcloud):
        array = np.frombuffer(pointcloud.data, dtype=np.float32)
        array = np.reshape(array, (-1, 4)).copy() # x, y, z, r
        return array # (N, 4) pointcloud

    
    def sync_callback(self, *msgs): # msgs is a tuple of whatever came in, in the same order you added to self.synchronizer_subs
        # self.get_logger().info(
        #     f"Got synchronized pair: RGB {image_rgb_msg.header.stamp.sec}.{image_rgb_msg.header.stamp.nanosec}, "
        #     f"Semseg {image_semseg_msg.header.stamp.sec}.{image_semseg_msg.header.stamp.nanosec}"
        # )
        
        rgb_msg = msgs[0]
        self.image_rgb = self.reshape_image(rgb_msg)
        lidar_msg = msgs[1]
        pointcloud = self.reshape_pointcloud(lidar_msg)
        self.pointcloud = pointcloud[:, :3]

        if self.compute_lanemarkings:
            semseg_msg = msgs[2]
            self.image_semseg = self.reshape_image(semseg_msg)
        if self.enable_pure_pursuit:
            depth_msg = msgs[3]
            self.image_depth = self.reshape_image(depth_msg)
        
        self.process_sensors()
        self.sync_done.set()
    

    def lanes_prediction_callback(self, msg):
        lanes_list_processed = [[] for _ in range(4)]
        lanes_list_processed[0] = [(int(point.x), int(point.y)) for point in msg.outer_left]
        lanes_list_processed[1] = [(int(point.x), int(point.y)) for point in msg.inner_left]
        lanes_list_processed[2] = [(int(point.x), int(point.y)) for point in msg.inner_right]
        lanes_list_processed[3] = [(int(point.x), int(point.y)) for point in msg.outer_right]
        self.lanes_list_processed = lanes_list_processed
        self.lane_prediction_done.set()


    def process_sensors(self):
        self.lanes_list_processed = []
        self.lidar_bboxes = []
        
        if self.compute_lanemarkings:
            ## Get current waypoints ### 
            waypoint = self.map.get_waypoint(self.ego_vehicle.get_location())
            waypoint_list = []
            for i in range(0, self.number_of_lanepoints):
                waypoint_list.append(waypoint.next(i + self.meters_per_frame)[0])

            lanes_list, x_lanes_list = self.lanemarkings.detect_lanemarkings(waypoint_list, self.image_semseg, self.camera_rgb)
            self.lanes_list_processed = self.lanemarkings.lanemarkings_processed(lanes_list)

        if self.compute_bbox:
            lidar_bboxes, labels = get_bboxes(self.world, self.pointcloud, self.lidar)
            self.lidar_bboxes = lidar_bboxes


    def run(self):
        
        self.render()


    def render(self):
        image_surface = pygame.surfarray.make_surface(self.image_rgb.swapaxes(0, 1)) # (W, H, C)
        self.display.blit(image_surface, (0, 0))

        if self.lanes_list_processed:
            # Draw lane on pygame window and binary mask
            inst_background = np.zeros_like(self.image_rgb)
            for i in range(len(self.lanes_list_processed)):
                for x, y, in self.lanes_list_processed[i]:
                    pygame.draw.circle(self.display, self.RGB_colors[i], (x, y), 3, 2)
                cv2.polylines(inst_background, np.int32([self.lanes_list_processed[i]]), isClosed=False, color=self.BGR_colors[i], thickness=5)                
            cv2.imshow("inst_background", inst_background)
            cv2.waitKey(1)


        # # Update point cloud
        # self.pcd.points = o3d.utility.Vector3dVector(self.pointcloud)
        # self.pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 1.0, 0.0], (self.pointcloud.shape[0], 1)))
        # self.vis.update_geometry(self.pcd)

        # # Clear previous bounding boxes
        # for line in self.bbox_lines:
        #     self.vis.remove_geometry(line, reset_bounding_box=False)
        # self.bbox_lines = []

        # bboxes_corners = bbox3d2corners(self.lidar_bboxes)
        # for corners in bboxes_corners:
        #     # Apply transformation to Open3D coordinate frame
        #     corners[:, 1] = -corners[:, 1] # convert from UE to Kitti/Open3D

        #     # Create LineSet
        #     line_set = o3d.geometry.LineSet()
        #     line_set.points = o3d.utility.Vector3dVector(corners)
        #     line_set.lines = o3d.utility.Vector2iVector(self.lines)

        #     # Set green color for all lines
        #     colors = [[0.0, 1.0, 0.0] for _ in range(len(self.lines))]
        #     line_set.colors = o3d.utility.Vector3dVector(colors)

        #     # Add to visualizer and keep reference
        #     self.vis.add_geometry(line_set)
        #     self.bbox_lines.append(line_set)

        # self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.cam)
        # self.vis.poll_events()
        # self.vis.update_renderer()
        # self.cam = self.vis.get_view_control().convert_to_pinhole_camera_parameters()


        pygame.display.flip()


    def destroy(self):
        self.get_logger().info('Cleaning up...')
        self.vehicle_manager.destroy()
        for sensor in self.sensors:
            sensor.destroy()
        self.get_logger().info('Sensors destroyed')

        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        self.get_logger().info('Synchronous mode disabled and actors destroyed.')

def main():
    rclpy.init()
    pygame.init()
    node = CarlaGame()

    try:
        # let carla ros setup topic first
        for _ in range(3):
            node.world.tick()

        while rclpy.ok():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                
            node.sync_done.clear()
            for event in node.modules_event:
                event.clear()

            ### Run simulation ###
            node.world.tick()

            # wait for sensor data
            while not node.sync_done.is_set():
                rclpy.spin_once(node, timeout_sec=0.1)

            node.process_sensors()

            # wait for other modules
            for event in node.modules_event:
                while not event.is_set():
                    rclpy.spin_once(node, timeout_sec=0.1)
                
            node.run()


    except Exception as e:
        node.get_logger().info('Shutting down...')
        traceback.print_exc()
    finally:
        # Clean shutdown
        node.destroy()
        node.destroy_node()
        rclpy.shutdown()
        pygame.quit()


if __name__ == '__main__':
    main()
