import random
import sys
import pygame
import cv2
import numpy as np

WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
BLUE = (100, 180, 255)
DARK_GRAY = (120, 120, 120)
MOVES = ["Rock", "Paper", "Scissors"]

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand vs Machine: Rock-Paper-Scissors")
FONT = pygame.font.SysFont("Arial", 28)
COUNTDOWN_FONT = pygame.font.SysFont("Arial", 72)
clock = pygame.time.Clock()
camera = cv2.VideoCapture(0)


class Button:
    """UI Button with position, label, and click detection."""
    def __init__(self, x, y, text):
        
        self.rect = pygame.Rect(x, y, 180, 60)
        self.text = text

    def draw(self, surface,mouse_pos):
        is_hovered = self.rect.collidepoint(mouse_pos)
        color = BLUE if is_hovered else GRAY
        pygame.draw.rect(surface, color, self.rect, border_radius=12)
        label = FONT.render(self.text, True, BLACK)
        surface.blit(label, label.get_rect(center=self.rect.center))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


class AI:
    """Simple AI player that makes random choices."""
    def __init__(self):
        self.history = []
        self.probs = [1/3, 1/3, 1/3] 

    def get_move(self):
        return random.choices(MOVES, self.probs, k=1)[0]

    def update_history(self, result):
        self.history.append(result)
        #TRAIN


class Game:
    """Game logic, score tracking, and UI management."""
    def __init__(self):
        self.player_score = 0
        self.ai_score = 0
        self.result_text = ""
        self.ai = AI()
        self.buttons = [
            Button(100, 450, "Rock"),
            Button(300, 450, "Paper"),
            Button(500, 450, "Scissors")
        ]

        # Countdown settings
        self.countdown_active = True
        self.countdown_start = pygame.time.get_ticks()
        self.countdown_length = 3000  # ms

    def update(self, surface):
        """Main drawing function for a game frame."""
        surface.fill(WHITE)
        self._draw_camera(surface)
        
        self._draw_score(surface)
        self._draw_result(surface)
        if self.countdown_active:
            self._draw_countdown(surface)
        else:
            self._draw_buttons(surface)

    def handle_click(self, pos):
        """Processes user click if countdown has finished."""
        if self.countdown_active:
            return
        for button in self.buttons:
            if button.is_clicked(pos):
                self._play_round(button.text)

    def _draw_camera(self, surface):
        """Grabs frame from webcam and displays it."""
        ret, frame = camera.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame_surface = pygame.surfarray.make_surface(frame)
            surface.blit(frame_surface, (0, 0))

    def _draw_buttons(self, surface):
        for button in self.buttons:
            button.draw(surface, pygame.mouse.get_pos())

    def _draw_score(self, surface):
        score = FONT.render(f"Score - You: {self.player_score} | AI: {self.ai_score}", True, BLACK)
        surface.blit(score, (100, 200))

    def _draw_result(self, surface):
        result = FONT.render(self.result_text, True, BLACK)
        surface.blit(result, (100, 150))

    def _draw_countdown(self, surface):
        """Displays animated countdown before game starts."""
        remaining = self.countdown_length + self.countdown_start - pygame.time.get_ticks()
        if remaining <= 0:
            self.countdown_active = False
            return

        number = str((remaining // 1000) + 1)
        scale = 1 + (remaining / self.countdown_length)
        countdown_surf = COUNTDOWN_FONT.render(number, True, BLACK)
        countdown_surf = pygame.transform.smoothscale(
            countdown_surf,
            (int(countdown_surf.get_width() * scale),
             int(countdown_surf.get_height() * scale))
        )
        surface.blit(
            countdown_surf,
            (WIDTH // 2 - countdown_surf.get_width() // 2,
             HEIGHT // 2 - countdown_surf.get_height() // 2)
        )

    def _play_round(self, player_move):
        ai_move = self.ai.get_move()
        result = self._determine_winner(player_move, ai_move)
        self.result_text = f"You: {player_move} | AI: {ai_move} → {result}"
        self.ai.update_history(result)
        self.countdown_start = pygame.time.get_ticks()
        self.countdown_active = True

    def _determine_winner(self, player, ai):
        """Game rules: determines outcome and updates score."""
        if player == ai:
            return "Draw"
        win_conditions = {
            "Rock": "Scissors",
            "Paper": "Rock",
            "Scissors": "Paper"
        }
        if win_conditions[player] == ai:
            self.player_score += 1
            return "You Win!"
        else:
            self.ai_score += 1
            return "AI Wins!"


# --------------------- MAIN LOOP ---------------------

def main():
    game = Game()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                camera.release()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                game.handle_click(event.pos)

        game.update(screen)
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
