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

    host = DeclareLaunchArgument('host', default_value='192.168.0.196')
    town = DeclareLaunchArgument('town', default_value='Town10HD_Opt')
    num_vehicles = DeclareLaunchArgument('num_vehicles', default_value='50')
    auto_run = DeclareLaunchArgument('auto_run', default_value='True')
    respawn = DeclareLaunchArgument('respawn', default_value='50')
    carla_auto_pilot = DeclareLaunchArgument('carla_auto_pilot', default_value='True')
    predict_lane = DeclareLaunchArgument('predict_lane', default_value='False')
    predict_object = DeclareLaunchArgument('predict_object', default_value='False')
    ld.add_action(host)
    ld.add_action(town)
    ld.add_action(num_vehicles)
    ld.add_action(auto_run)
    ld.add_action(respawn)
    ld.add_action(carla_auto_pilot)
    ld.add_action(predict_lane)
    ld.add_action(predict_object)

    carla_ros_bridge_launch_file = os.path.join(
        get_package_share_directory('carla_ros_bridge'),
        'carla_ros_bridge.launch.py'
    )

    carla_ros_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(carla_ros_bridge_launch_file),
        launch_arguments={
            'host': LaunchConfiguration('host'),
            'synchronous_mode': 'True',
            'passive': 'True',
        }.items(),
    )
    ld.add_action(carla_ros_bridge_launch)

    carla_game_node = Node(
        package="carla_client",
        executable="carla_game",
        parameters=[
            {"host": LaunchConfiguration('host')},
            {"town": LaunchConfiguration('town')},
            {"num_vehicles": LaunchConfiguration('num_vehicles')},
            {"auto_run": LaunchConfiguration('auto_run')},
            {"respawn": LaunchConfiguration('respawn')},
            {"carla_auto_pilot": LaunchConfiguration('carla_auto_pilot')},
            {"predict_lane": LaunchConfiguration('predict_lane')},
            {"predict_object": LaunchConfiguration('predict_object')},
        ]
    )
    ld.add_action(carla_game_node)

    return ld