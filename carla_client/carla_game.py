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

from sensor_msgs.msg import Image

from carla_client.vehicle_manager import VehicleManager
from carla_client.lanemarkings import LaneMarkings

class CarlaGame(Node):
    def __init__(self):
        super().__init__('carla_game')

        # Get parameters
        self.host = self.declare_parameter("host", "192.168.0.196").value
        self.town = self.declare_parameter("town", "Town10HD_Opt").value
        self.num_vehicles = self.declare_parameter("num_vehicles", 50).value
        self.respawn = self.declare_parameter("respawn", 50).value
        self.carla_auto_pilot = self.declare_parameter("carla_auto_pilot", True).value
        self.predict_lane = self.declare_parameter("predict_lane", False).value
        self.predict_object = self.declare_parameter("predict_object", False).value

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

        if not self.predict_lane:
            # Spawn semseg-cam and attach to vehicle
            bp_camera_semseg = blueprint_library.find('sensor.camera.semantic_segmentation')
            bp_camera_semseg.set_attribute('image_size_x', f'{self.image_width}')
            bp_camera_semseg.set_attribute('image_size_y', f'{self.image_height}')
            bp_camera_semseg.set_attribute('fov', f'{self.fov}')
            bp_camera_semseg.set_attribute('role_name', 'semseg')
            self.camera_semseg = self.world.spawn_actor(bp_camera_semseg, self.camera_spawnpoint, attach_to=self.ego_vehicle)
            self.sensors.append(self.camera_semseg)
        else:
            self.camera_semseg = None

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

        if self.predict_lane:
            # self.lanedet = LaneDet()
            pass
        else:
            self.lanemarkings = LaneMarkings(self.world, self.image_width, self.image_height, self.fov, self.number_of_lanepoints, self.meters_per_frame)


        # Subscribers
        # self.camera_rgb_sub = self.create_subscription(Image, '/carla/hero/rgb/image', self.camera_rgb_callback, 10)
        # self.camera_semseg_sub = self.create_subscription(Image, '/carla/hero/semseg/image', self.camera_semseg_callback, 10)
        self.camera_rgb_sub = message_filters.Subscriber(self, Image, '/carla/hero/rgb/image')
        self.camera_semseg_sub = message_filters.Subscriber(self, Image, '/carla/hero/semseg/image')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.camera_rgb_sub, self.camera_semseg_sub],
            queue_size=10,
            slop=0.05   # allowed timestamp difference in seconds
        )
        self.ts.registerCallback(self.sync_callback)

        self.sync_done = threading.Event()

    def reshape_image(self, image):
        array = np.frombuffer(image.data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4)) # BGRA
        array = array[:, :, :3] # BGR
        array = array[:, :, ::-1] # RGB, (H, W, C)
        return array # (H, W, C)
    
    
    def camera_rgb_callback(self, image):
        self.image_rgb = self.reshape_image(image)


    def camera_semseg_callback(self, image):
        self.image_semseg = self.reshape_image(image)


    def sync_callback(self, image_rgb_msg, image_semseg_msg):
        # self.get_logger().info(
        #     f"Got synchronized pair: RGB {image_rgb_msg.header.stamp.sec}.{image_rgb_msg.header.stamp.nanosec}, "
        #     f"Semseg {image_semseg_msg.header.stamp.sec}.{image_semseg_msg.header.stamp.nanosec}"
        # )

        self.image_rgb = self.reshape_image(image_rgb_msg)
        self.image_semseg = self.reshape_image(image_semseg_msg)
        self.run()


        self.sync_done.set()

    def run(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            
        ## Get current waypoints ### 
        waypoint = self.map.get_waypoint(self.ego_vehicle.get_location())
        waypoint_list = []
        for i in range(0, self.number_of_lanepoints):
            waypoint_list.append(waypoint.next(i + self.meters_per_frame)[0])

        ### Predict lanepoints for all lanes ###
        if self.predict_lane:
            # img = Image.fromarray(image_rgb, mode="RGB") 
            # lanes_list_processed = self.lanedet.predict(img)
            pass
        else:
            lanes_list, x_lanes_list = self.lanemarkings.detect_lanemarkings(waypoint_list, self.image_semseg, self.camera_rgb)
            lanes_list_processed = self.lanemarkings.lanemarkings_processed(lanes_list)
    
        self.render(lanes_list_processed)


    def render(self, lanes_list_processed):
        image_surface = pygame.surfarray.make_surface(self.image_rgb.swapaxes(0, 1)) # (W, H, C)
        self.display.blit(image_surface, (0, 0))

        inst_background = np.zeros_like(self.image_rgb)

        # Draw lane on pygame window and binary mask
        if(self.render_lanes):
            for i in range(len(lanes_list_processed)):
                for x, y, in lanes_list_processed[i]:
                    pygame.draw.circle(self.display, self.RGB_colors[i], (x, y), 3, 2)
                cv2.polylines(inst_background, np.int32([lanes_list_processed[i]]), isClosed=False, color=self.BGR_colors[i], thickness=5)                

        pygame.display.flip()
        cv2.imshow("inst_background", inst_background)
        cv2.waitKey(1)


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

    node.get_logger().info('OK')

    try:
        # let carla ros setup topic first
        for _ in range(3):
            node.world.tick()

        while rclpy.ok():

            node.sync_done.clear()

            ### Run simulation ###
            node.world.tick()

            while not node.sync_done.is_set():
                rclpy.spin_once(node, timeout_sec=0.1)


    except Exception as e:
        node.get_logger().info('Shutting down...')
        node.get_logger().info(e)
    finally:
        # Clean shutdown
        node.destroy()
        node.destroy_node()
        rclpy.shutdown()
        pygame.quit()


if __name__ == '__main__':
    main()
