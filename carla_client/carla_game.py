#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import carla
import random
import pygame
import time

from carla_client.vehicle_manager import VehicleManager

class CarlaGame(Node):
    def __init__(self):
        super().__init__('carla_game')

        # Get parameters
        self.town = self.declare_parameter("town", "Town10HD_Opt").value
        self.num_vehicles = self.declare_parameter("num_vehicles", 50).value
        self.respawn = self.declare_parameter("respawn", 50).value
        self.carla_auto_pilot = self.declare_parameter("carla_auto_pilot", True).value
        self.predict_lane = self.declare_parameter("predict_lane", False).value
        self.predict_object = self.declare_parameter("predict_object", False).value

        self.weather = carla.WeatherParameters.ClearNoon
        self.fps = 10
        self.image_width = 1280
        self.image_height = 720
        self.fov = 90

        self.display = pygame.display.set_mode((self.image_width, self.image_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pygame_clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 36) 

        # Connect client
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(self.town)
        self.world.set_weather(self.weather)
        
        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 10
        self.world.apply_settings(settings)

        self.map = self.world.get_map()

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
        self.camera_spawnpoint = carla.Transform(carla.Location(x=1.0, z=2.0), carla.Rotation(pitch=-18.5)) # camera 5
        self.camera_rgb = self.world.spawn_actor(bp_camera_rgb, self.camera_spawnpoint, attach_to=self.ego_vehicle)
        self.sensors.append(self.camera_rgb)

        if not self.predict_lane:
            # Spawn semseg-cam and attach to vehicle
            bp_camera_semseg = blueprint_library.find('sensor.camera.semantic_segmentation')
            bp_camera_semseg.set_attribute('image_size_x', f'{self.image_width}')
            bp_camera_semseg.set_attribute('image_size_y', f'{self.image_height}')
            bp_camera_semseg.set_attribute('fov', f'{self.fov}')
            self.camera_semseg = self.world.spawn_actor(bp_camera_semseg, self.camera_spawnpoint, attach_to=self.ego_vehicle)
            self.sensors.append(self.camera_semseg)
        else:
            self.camera_semseg = None

        # Spawn depth-cam and attach to vehicle
        bp_camera_depth = blueprint_library.find('sensor.camera.depth')
        bp_camera_depth.set_attribute('image_size_x', f'{self.image_width}')
        bp_camera_depth.set_attribute('image_size_y', f'{self.image_height}')
        bp_camera_depth.set_attribute('fov', f'{self.fov}')
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

    def run(self):
        self.world.tick()  # Advance one simulation step
        self.get_logger().info('Tick')
    

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
        while rclpy.ok():
            # Process callbacks once (non-blocking)
            rclpy.spin_once(node, timeout_sec=0.01)

            # Custom logic — tick the CARLA world
            node.run()

            # Optional sleep to avoid maxing CPU
            # time.sleep(0.01)

    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        # Clean shutdown
        node.destroy()
        node.destroy_node()
        rclpy.shutdown()
        pygame.quit()


if __name__ == '__main__':
    main()
