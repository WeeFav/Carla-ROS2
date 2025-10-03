#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 200, 0)

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller')

        # Fixed parameters
        self.hud_width = 600
        self.hud_height = 400
        
        # Set up pygame
        self.display = pygame.display.set_mode((self.hud_width, self.hud_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pygame_clock = pygame.time.Clock()
        self.big_font = pygame.font.SysFont("Arial", 28)
        self.small_font = pygame.font.SysFont("Arial", 20)


    def draw_bar(self, x, y, w, h, value, color, label, center_zero=False):
        """Draws a bar for throttle/brake/steer."""
        pygame.draw.rect(self.display, WHITE, (x, y, w, h), 2)

        if center_zero:  # For steering: negative=left, positive=right
            mid = x + w // 2
            pygame.draw.line(self.display, WHITE, (mid, y), (mid, y + h), 2)
            if value >= 0:
                fill = int((w // 2) * value)
                pygame.draw.rect(self.display, color, (mid, y, fill, h))
            else:
                fill = int((w // 2) * -value)
                pygame.draw.rect(self.display, color, (mid - fill, y, fill, h))
        else:
            fill = int(w * value)
            pygame.draw.rect(self.display, color, (x, y, fill, h))

        text = self.small_font.render(f"{label}: {value:.2f}", True, WHITE)
        self.display.blit(text, (x, y - 20))

    
    def draw_hud(self):
        self.display.fill(BLACK)

        # Draw control bars
        self.draw_bar(50, 80, 200, 25, throttle, GREEN, "Throttle")
        self.draw_bar(50, 150, 200, 25, brake, RED, "Brake")
        self.draw_bar(50, 220, 200, 25, steer, BLUE, "Steer", center_zero=True)

        # Speed display
        speed_text = self.big_font.render(f"Speed: {speed:.1f} km/h", True, YELLOW)
        self.display.blit(speed_text, (320, 100))

        # Mode display
        mode_color = GREEN if mode == "AUTO" else RED
        mode_text = self.big_font.render(f"Mode: {mode}", True, mode_color)
        self.display.blit(mode_text, (320, 160))

        pygame.display.flip()
    

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key == K_b:


def main():
    rclpy.init()
    pygame.init()
    node = MainController()

    while rclpy.ok():
        node.parse_events()
        pygame.quit()
        # Example: Update values here (in real use, read from your car controller)
        # throttle, steer, brake, speed, mode = get_car_state()

        draw_hud()

    node.destroy_node()
    rclpy.shutdown()
    pygame.quit()


if __name__ == '__main__':
    main()