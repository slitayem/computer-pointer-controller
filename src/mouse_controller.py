"""
@author: Saloua Litayem
Mouse controller class
"""
import pyautogui


class MouseController:
    """
    Helper class to control the mouse pointer.
    """
    def __init__(self, mouse_precision=12, mouse_speed=0.05):
        self.precision = mouse_precision
        self.speed = mouse_speed

    def center(self):
        """Move the mouse cursor to the center of the screen"""
        screen_size = pyautogui.size()
        pyautogui.moveTo(screen_size.width // 2, screen_size.height // 2)

    def move_to(self, x_pos, y_pos):
        """Move the mouse cursor to the position x,y"""
        pyautogui.moveTo(x_pos, y_pos)

    def move(self, x_pos, y_pos):
        """Move the mouse controller"""
        pyautogui.moveRel(x_pos * self.precision, -1 * y_pos * self.precision,
            duration=self.speed)
