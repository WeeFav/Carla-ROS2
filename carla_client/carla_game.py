#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import carla
import random
import pygame
import time

class CarlaGame(Node):
    def __init__(self):
        super().__init__('carla_game')

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town10HD_Opt')
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 20
        self.world.apply_settings(settings)
        # self.map = self.world.get_map()
        # self.world.set_weather(cfg.weather)

        self.tm = self.client.get_trafficmanager()
        self.tm.set_synchronous_mode(True)

        self.spawn_ego_vehicle()
        self.attach_camera()

        self.get_logger().info('Ego vehicle with camera spawned in synchronous mode.')


    def spawn_ego_vehicle(self):
        blueprint_library = self.world.get_blueprint_library()
        bp_vehicle = blueprint_library.find('vehicle.ford.mustang')
        bp_vehicle.set_attribute('role_name', 'ego_vehicle')

        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.ego_vehicle = self.world.spawn_actor(bp_vehicle, spawn_point)
        self.ego_vehicle.set_autopilot(True)
        self.get_logger().info(f'Ego vehicle spawned: {self.ego_vehicle.id}')

    def attach_camera(self):
        blueprint_library = self.world.get_blueprint_library()
        # Spawn rgb-cam and attach to vehicle
        bp_camera_rgb = blueprint_library.find('sensor.camera.rgb')
        bp_camera_rgb.set_attribute('image_size_x', f'1280')
        bp_camera_rgb.set_attribute('image_size_y', f'720')
        bp_camera_rgb.set_attribute('fov', f'90')
        self.camera_spawnpoint = carla.Transform(carla.Location(x=1.0, z=2.0), carla.Rotation(pitch=-18.5)) # camera 5
        self.camera_rgb = self.world.spawn_actor(bp_camera_rgb, self.camera_spawnpoint, attach_to=self.ego_vehicle)

    def run(self):
        self.world.tick()  # Advance one simulation step
        self.get_logger().info('World ticked.')

    def destroy(self):
        self.get_logger().info('Cleaning up...')
        if self.camera_rgb:
            self.camera_rgb.stop()
            self.camera_rgb.destroy()
        if self.ego_vehicle:
            self.ego_vehicle.destroy()

        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        self.get_logger().info('Synchronous mode disabled and actors destroyed.')

def main():
    rclpy.init()
    node = CarlaGame()

    try:
        while rclpy.ok():
            # Process callbacks once (non-blocking)
            rclpy.spin_once(node, timeout_sec=0.01)

            # Custom logic — tick the CARLA world
            node.run()

            # Optional sleep to avoid maxing CPU
            time.sleep(3)

    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        # Clean shutdown
        node.destroy()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
