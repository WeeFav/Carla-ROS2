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

    pkg_share = get_package_share_directory('carla_client')

    model_path = DeclareLaunchArgument('model_path', default_value=os.path.join(pkg_share, 'ep049.pth'))
    use_classification = DeclareLaunchArgument('use_classification', default_value='True')
    ld.add_action(model_path)
    ld.add_action(use_classification)

    lanedet_node = Node(
        package="carla_client",
        executable="lanedet",
        parameters=[
            {"model_path": LaunchConfiguration('model_path')},
            {"use_classification": LaunchConfiguration('use_classification')},
        ]
    )
    ld.add_action(lanedet_node)

    return ld