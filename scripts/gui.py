import pygame
import sys

# --- Constants ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
CONTROL_PANEL_WIDTH = 300
GAME_PANEL_WIDTH = SCREEN_WIDTH - CONTROL_PANEL_WIDTH

BG_COLOR = (240, 240, 240)
TEXT_COLOR = (0, 0, 0)
INPUT_BOX_COLOR = (255, 255, 255)
BUTTON_COLOR = (100, 100, 200)
BUTTON_TEXT_COLOR = (255, 255, 255)

# --- InputBox Class ---
class InputBox:
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = INPUT_BOX_COLOR
        self.text = text
        self.font = pygame.font.Font(None, 32)
        self.txt_surface = self.font.render(text, True, self.color)
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    self.active = False
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                self.txt_surface = self.font.render(self.text, True, TEXT_COLOR)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        pygame.draw.rect(screen, TEXT_COLOR, self.rect, 2)

# --- Button Class ---
class Button:
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = BUTTON_COLOR
        self.text = text
        self.font = pygame.font.Font(None, 32)
        self.txt_surface = self.font.render(text, True, BUTTON_TEXT_COLOR)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        screen.blit(self.txt_surface, (self.rect.x + (self.rect.w - self.txt_surface.get_width()) // 2, self.rect.y + (self.rect.h - self.txt_surface.get_height()) // 2))

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

# --- Main Application Class ---
class SlimeVolleyGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Slime Volley AI Lab for Kids")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 28)

        self.input_boxes = {
            "jump_threshold": InputBox(50, 100, 200, 32, "0.5"),
            "left_threshold": InputBox(50, 180, 200, 32, "-0.5"),
            "right_threshold": InputBox(50, 260, 200, 32, "0.5"),
        }

        self.buttons = {
            "train": Button(50, 340, 200, 40, "Train Agent"),
            "play": Button(50, 400, 200, 40, "Play Game"),
            "play_vs_agent": Button(50, 460, 200, 40, "Play vs Agent"),
        }

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                for box in self.input_boxes.values():
                    box.handle_event(event)
                for name, button in self.buttons.items():
                    if button.is_clicked(event):
                        self.handle_button_click(name)

            self.draw()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

import json

# --- SimpleRuleBasedAgent Class ---
class SimpleRuleBasedAgent:
    def __init__(self, jump_threshold=0.5, left_threshold=-0.5, right_threshold=0.5):
        self.jump_threshold = jump_threshold
        self.left_threshold = left_threshold
        self.right_threshold = right_threshold

    def predict(self, obs):
        action = [0, 0, 0]
        if obs[5] > self.jump_threshold:
            action[2] = 1
        if obs[4] < self.left_threshold:
            action[0] = 1
        if obs[4] > self.right_threshold:
            action[1] = 1
        return action

# --- Main Application Class ---
class SlimeVolleyGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Slime Volley AI Lab for Kids")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 28)

        self.input_boxes = {
            "jump_threshold": InputBox(50, 100, 200, 32, "0.5"),
            "left_threshold": InputBox(50, 180, 200, 32, "-0.5"),
            "right_threshold": InputBox(50, 260, 200, 32, "0.5"),
        }

        self.buttons = {
            "train": Button(50, 340, 200, 40, "Train Agent"),
            "play": Button(50, 400, 200, 40, "Play Game"),
            "play_vs_agent": Button(50, 460, 200, 40, "Play vs Agent"),
        }
        self.agent = None

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                for box in self.input_boxes.values():
                    box.handle_event(event)
                for name, button in self.buttons.items():
                    if button.is_clicked(event):
                        self.handle_button_click(name)

            self.draw()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

    def handle_button_click(self, name):
        if name == "train":
            try:
                jump_threshold = float(self.input_boxes["jump_threshold"].text)
                left_threshold = float(self.input_boxes["left_threshold"].text)
                right_threshold = float(self.input_boxes["right_threshold"].text)
                
                self.agent = SimpleRuleBasedAgent(jump_threshold, left_threshold, right_threshold)
                
                agent_params = {
                    "jump_threshold": jump_threshold,
                    "left_threshold": left_threshold,
                    "right_threshold": right_threshold,
                }
                
                with open("my_agent.json", "w") as f:
                    json.dump(agent_params, f)
                
                print("Agent trained and saved!")
            except ValueError:
                print("Invalid input! Please enter numbers for the thresholds.")
        else:
            print(f"Button {name} clicked")

    def draw(self):
        self.screen.fill(BG_COLOR)
        self.draw_control_panel()
        self.draw_game_panel()
        pygame.display.flip()

    def draw_control_panel(self):
        pygame.draw.rect(self.screen, (200, 200, 200), (0, 0, CONTROL_PANEL_WIDTH, SCREEN_HEIGHT))
        title_text = self.font.render("AI Agent Creator", True, TEXT_COLOR)
        self.screen.blit(title_text, (20, 20))

        # Draw input boxes and labels
        jump_label = self.small_font.render("Jump if ball is higher than:", True, TEXT_COLOR)
        self.screen.blit(jump_label, (50, 70))
        self.input_boxes["jump_threshold"].draw(self.screen)

        left_label = self.small_font.render("Move left if ball is left of:", True, TEXT_COLOR)
        self.screen.blit(left_label, (50, 150))
        self.input_boxes["left_threshold"].draw(self.screen)

        right_label = self.small_font.render("Move right if ball is right of:", True, TEXT_COLOR)
        self.screen.blit(right_label, (50, 230))
        self.input_boxes["right_threshold"].draw(self.screen)

        # Draw buttons
        for button in self.buttons.values():
            button.draw(self.screen)

    def draw_game_panel(self):
        pygame.draw.rect(self.screen, (0, 0, 0), (CONTROL_PANEL_WIDTH, 0, GAME_PANEL_WIDTH, SCREEN_HEIGHT))
        game_text = self.font.render("Game will be displayed here", True, (255, 255, 255))
        self.screen.blit(game_text, (CONTROL_PANEL_WIDTH + 50, SCREEN_HEIGHT // 2))

if __name__ == "__main__":
    app = SlimeVolleyGUI()
    app.run()
