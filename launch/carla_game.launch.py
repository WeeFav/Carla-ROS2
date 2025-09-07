from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import carla

def generate_launch_description():
    ld = LaunchDescription()

    carla_game_node = Node(
        package="carla_client",
        executable="carla_game",
        parameters=[
            {"town": "Town10HD_Opt"},
            {"num_vehicles": 50},
            {"auto_run": True},
            {"respawn": 50},
            {"carla_auto_pilot": True},
            {"predict_lane": False},
            {"predict_object": False},
        ]
    )

    ld.add_action(carla_game_node)

    return ld