cmake_minimum_required(VERSION 3.5)
project(carla_client)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)   # Example dependency for your .msg

find_package(rosidl_default_generators REQUIRED)

# -----------------------------
# 1. Generate custom messages
# -----------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Person.msg"
  DEPENDENCIES std_msgs
)

# -----------------------------
# 2. Install Python package
# -----------------------------
ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/${PROJECT_NAME})

# -----------------------------
# 3. Install Python executables
# -----------------------------
install(PROGRAMS
  src/${PROJECT_NAME}/my_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# -----------------------------
ament_package()
