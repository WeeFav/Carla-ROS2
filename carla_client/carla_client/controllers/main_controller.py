#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import pygame

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller')

    

def main():
    rclpy.init()
    pygame.init()
    node = MainController()



    node.destroy_node()
    rclpy.shutdown()
    pygame.quit()


if __name__ == '__main__':
    main()