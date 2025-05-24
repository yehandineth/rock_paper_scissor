import random
import sys
import pygame
import cv2
import numpy as np
import mediapipe as mp
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io


WIDTH, HEIGHT = 1280, 720
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

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
gesture_result = {"gesture": "Unknown"}

def gesture_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        finger_states = get_finger_states(landmarks)
        gesture_result["gesture"] = classify_hand_gesture(finger_states)
    else:
        gesture_result["gesture"] = "Unknown"

def get_finger_states(landmarks):
    tips = [4, 8, 12, 16, 20]
    pip_joints = [2, 6, 10, 14, 18]
    finger_states = []

    # Thumb special case: compare x-coordinates
    if landmarks[tips[0]].x < landmarks[pip_joints[0]].x:
        finger_states.append(1)
    else:
        finger_states.append(0)

    # Other fingers: compare y-coordinates
    for i in range(1, 5):
        finger_states.append(1 if landmarks[tips[i]].y < landmarks[pip_joints[i]].y else 0)

    return finger_states

def classify_hand_gesture(finger_states):
    if finger_states == [0, 0, 0, 0, 0]:
        return "Rock"
    elif finger_states == [1, 1, 1, 1, 1]:
        return "Paper"
    elif finger_states == [0, 1, 1, 0, 0]:
        return "Scissors"
    else:
        return "Unknown"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=gesture_callback
)

landmarker = HandLandmarker.create_from_options(options)


    

class Button:
    """UI Button with position, label, and click detection."""
    def __init__(self, x, y, text):
        
        self.rect = pygame.Rect(x, y, 180, 60)
        self.text = text
        self.Toggle = False

    def draw(self, surface,mouse_pos):
        is_hovered = self.rect.collidepoint(mouse_pos)
        color = BLUE if (is_hovered or self.Toggle) else GRAY
        pygame.draw.rect(surface, color, self.rect, border_radius=12)
        label = FONT.render(self.text, True, BLACK)
        surface.blit(label, label.get_rect(center=self.rect.center))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


class AI:
    """Simple AI player that makes random choices."""
    def __init__(self):
        self.history = []
        self.transition_matrix = np.ones(shape=(3,3,3))

    def get_move(self):
        probs = self.transition_matrix[self.history[-2],self.history[-1]]/ self.transition_matrix[self.history[-2],self.history[-1]].sum() if len(self.history) > 2 else np.array([1/3, 1/3, 1/3])
        result = random.choices([0,1,2], probs, k=1)[0]
        return MOVES[result]

    def update_history(self, result):
        self.history.append(result)

        # update the transition matrix
        if len(self.history) > 2:
            self.transition_matrix[self.history[-3],self.history[-2],(self.history[-1]+1)%3] += 1
            print(self.transition_matrix)
        #TRAIN
   

class Game:
    """Game logic, score tracking, and UI management."""
    def __init__(self):
        self.mode = 'PLAY'
        self.player_score = 0
        self.ai_score = 0
        self.result_text = ""
        self.ai = AI()
        self.buttons = [
            Button(100, 650, "Toggle Insights"),
            # Button(500, 600, "Paper"),
            Button(900, 650, "Toggle Video")
        ]
        self.buttons[1].Toggle = True

        # Countdown settings
        self.countdown_active = True
        self.countdown_start = pygame.time.get_ticks()
        self.countdown_length = 3000  # ms

    def update(self, surface):
        """Main drawing function for a game frame."""
        surface.fill(WHITE)
        self._draw_camera(surface)
        self._draw_buttons(surface)
        
        self._draw_result_rect(surface) if not self.buttons[0].Toggle else None
        if self.countdown_active and not self.buttons[0].Toggle:
            self._draw_countdown(surface)
        if self.countdown_active == False and gesture_result["gesture"] != "Unknown" and not self.buttons[0].Toggle:
            self._play_round(gesture_result["gesture"])
            gesture_result["gesture"] = "Unknown"
        
            

    def handle_click(self, pos):
        """Processes user click if countdown has finished."""
        for button in self.buttons:
            if button.is_clicked(pos):
                if button.text == "Toggle Video":
                    button.Toggle = not button.Toggle
                else:
                    button.Toggle = not button.Toggle

    def _draw_camera(self, surface):
        ret, frame = camera.read()
        if ret and not self.buttons[0].Toggle:
            frame = cv2.flip(frame, 1)  # Mirror the image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Current timestamp in milliseconds
            timestamp_ms = int(pygame.time.get_ticks())

            landmarker.detect_async(mp_image, timestamp_ms)

            # Display detected gesture from global dictionary
            gesture_text = FONT.render(f"Gesture: {gesture_result['gesture']}", True, BLACK)
            frame_surface = pygame.surfarray.make_surface(np.rot90(rgb_frame))
            surface.blit(frame_surface, (75, 150)) if self.buttons[1].Toggle else None
            surface.blit(gesture_text, (WIDTH - 250,50))
        else:

            matrix = self.ai.transition_matrix

            fig, axes = plt.subplots(3, 3, figsize=(12, 9))

            moves = ['Rock', 'Paper', 'Scissors']

            for i in range(3):
                for j in range(3):
                    data = matrix[i, j].copy()
                    data[:2],data[2] = data[1:],data[0]
                    normalized_data = data 
                    axes[i, j].bar(moves, normalized_data, color=[plt.cm.viridis(x/np.max(matrix)) for x in normalized_data])
                    axes[i, j].set_ylim(0, np.max(data) +1)
                    axes[i, j].set_title(f'Sequence: {moves[i]} → {moves[j]}', fontsize=14, fontweight='bold')
                    axes[i, j].set_ylabel('Probability')

            plt.suptitle('How you think.', fontsize=20, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)

            img = pygame.image.load(buf, 'buffer')
            img = pygame.transform.scale(img, (1000, 550))
            surface.blit(img, ((WIDTH - 1000) // 2, (HEIGHT - 550) // 2))


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
