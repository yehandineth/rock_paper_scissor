import random
import sys
import pygame
import cv2
import numpy as np
# import mediapipe as mp

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
        self.probs = np.array([1/3, 1/3, 1/3]) 
        self.transition_matrix = np.ones(shape=(3,3)) / 3
        self.trans_temps = np.array([[3],[3],[3]])

    def get_move(self):
        probs = self.probs * self.transition_matrix[self.history[-1]] if len(self.history) > 0 else self.probs
        result = random.choices([0,1,2], probs, k=1)[0]
        return MOVES[result]

    def update_history(self, result):
        tmp = self.probs * (len(self.history) + 3)
        self.history.append(result)
        tmp[(result + 1) % 3] += 1
        self.probs = tmp / np.sum(tmp)

        # update the transition matrix
        if len(self.history) >= 2:
            self.transition_matrix[self.history[-2]] *= self.trans_temps[self.history[-2]] 
            self.transition_matrix[self.history[-2],(self.history[-1]+1)%3] += 1
            self.trans_temps[self.history[-2]] += 1
            self.transition_matrix[self.history[-2]] /= self.trans_temps[self.history[-2]]
            print(self.transition_matrix)
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
        
        self._draw_result_rect(surface) 
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

        
    def _draw_result_rect(self, surface):
        x = 50
        y = 50
        result_rect = pygame.Rect(x - 20, y - 20, 700, 100)
        pygame.draw.rect(surface, DARK_GRAY, result_rect, border_radius=12)
        score = FONT.render(f"Score - You: {self.player_score} | AI: {self.ai_score}", True, BLACK)
        result = FONT.render(self.result_text, True, BLACK)
        
        result = FONT.render(self.result_text, True, BLACK)
        surface.blit(result, (x-5, y))
        surface.blit(score, (x, y + 30))


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
        self.ai.update_history(MOVES.index(player_move))

        self.result_text = f"You: {player_move} | AI: {ai_move} → {result}"
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

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
# mp_draw = mp.solutions.drawing_utils

def get_finger_states(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    pip_joints = [2, 6, 10, 14, 18]
    finger_states = []

    # Thumb: special case (check x instead of y)
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[pip_joints[0]].x:
        finger_states.append(1)
    else:
        finger_states.append(0)

    # Other fingers: tip higher than joint → extended
    for i in range(1, 5):
        tip = hand_landmarks.landmark[tips[i]]
        pip = hand_landmarks.landmark[pip_joints[i]]
        finger_states.append(1 if tip.y < pip.y else 0)

    return finger_states

def classify_hand_gesture(finger_states):
    if finger_states == [0, 0, 0, 0, 0]:
        return "Rock"
    elif finger_states == [1, 1, 1, 1, 1]:
        return "Paper"
    elif finger_states[1] and finger_states[2] and not any(finger_states[3:]):
        return "Scissors"
    else:
        return "Unknown"


def main():
    game = Game()
    print('RUNNING')
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
    print('RUNNING')

    main()
