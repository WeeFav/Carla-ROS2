from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    ld = LaunchDescription()

    lanedet_node = Node(
        package="carla_client",
        executable="lanedet",
        output='screen',
        emulate_tty=True,
    )
    ld.add_action(lanedet_node)

    return ld