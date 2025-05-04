import random
import sys
import pygame

print(sys.executable)
# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
MOVES = ["Rock", "Paper", "Scissors"]

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand vs Machine: Rock-Paper-Scissors")
font = pygame.font.SysFont("Arial", 28)


class Button:
    def __init__(self, x, y, text):
        self.rect = pygame.Rect(x, y, 200, 60)
        self.text = text

    def draw(self, surface):
        pygame.draw.rect(surface, GRAY, self.rect)
        label = font.render(self.text, True, BLACK)
        surface.blit(label, (self.rect.x + 20, self.rect.y + 15))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


class Game:
    
    def __init__(self):
        self.player_score = 0
        self.ai_score = 0
        self.result_text = ""
        self.buttons = [
            Button(100, 450, "Rock"),
            Button(300, 450, "Paper"),
            Button(500, 450, "Scissors")
        ]

    def get_ai_move(self):
        return random.choice(MOVES)

    def decide_winner(self, player, ai):
        if player == ai:
            return "Draw"
        elif (player == "Rock" and ai == "Scissors") or \
             (player == "Paper" and ai == "Rock") or \
             (player == "Scissors" and ai == "Paper"):
            self.player_score += 1
            return "You Win!"
        else:
            self.ai_score += 1
            return "AI Wins!"

    def handle_click(self, pos):
        for button in self.buttons:
            if button.is_clicked(pos):
                player_move = button.text
                ai_move = self.get_ai_move()
                result = self.decide_winner(player_move, ai_move)
                self.result_text = f"You: {player_move} | AI: {ai_move} → {result}"

    def draw(self, surface):
        surface.fill(WHITE)
        for button in self.buttons:
            button.draw(surface)

        result_surface = font.render(self.result_text, True, BLACK)
        score_surface = font.render(f"Score - You: {self.player_score} | AI: {self.ai_score}", True, BLACK)

        surface.blit(result_surface, (100, 150))
        surface.blit(score_surface, (100, 200))

class AI:

    def __init__(self):
        self.probs = [0,0,0]

    def get_move(self):
        return random.choice(MOVES, self.probs)

    def update_score(self, result):
        if result == "AI Wins!":
            self.score += 1

def main():
    clock = pygame.time.Clock()
    game = Game()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                game.handle_click(event.pos)

        game.draw(screen)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
