from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

import carla

def generate_launch_description():
    ld = LaunchDescription()

    carla_ros_bridge_launch_file = os.path.join(
        get_package_share_directory('carla_ros_bridge'),
        'carla_ros_bridge.launch.py'
    )

    carla_ros_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(carla_ros_bridge_launch_file),
        launch_arguments={
            'host': '192.168.0.196',
            'synchronous_mode': 'True',
            'passive': 'True',
        }.items(),
    )
    ld.add_action(carla_ros_bridge_launch)

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