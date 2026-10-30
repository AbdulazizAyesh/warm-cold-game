"""
Warm & Cold Game - a hands-free number-guessing game controlled by your fingers.

A webcam tracks your hands with MediaPipe. You "guess" a hidden target number
(1-10) by holding up that many fingers; the game replies with Warm/Cold hints and
chooses the next target with a small Markov (MDP) model, so the challenge scales
with the selected difficulty.

Controls: click START, hold up fingers to pick a difficulty (1/2/3), hold a finger
count steady for a moment to lock in a guess, and press Q to quit.
"""

import random
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound

# --- Configuration ---
CAMERA_INDEX = 0        # webcam index (set to 1 if the wrong camera opens)
HOLD_SECONDS = 2        # seconds a finger count must stay steady to count as a guess
GRACE_PERIOD = 2        # seconds after a round starts before guesses are accepted
SUCCESS_SOUND = "success.mp3"
GAMEOVER_SOUND = "gameover.mp3"

# --- MediaPipe hand tracking ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)


def count_fingers(hand_landmarks, hand_label):
    """Return how many fingers are raised on a single detected hand."""
    finger_tips = [mp_hands.HandLandmark.THUMB_TIP,
                   mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]

    finger_pips = [mp_hands.HandLandmark.THUMB_IP,
                   mp_hands.HandLandmark.INDEX_FINGER_PIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                   mp_hands.HandLandmark.RING_FINGER_PIP,
                   mp_hands.HandLandmark.PINKY_PIP]

    fingers = []

    for i in range(1, 5):  # Check fingers (index to pinky)
        if hand_landmarks.landmark[finger_tips[i]].y < hand_landmarks.landmark[finger_pips[i]].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Check thumb separately
    if hand_label == "Right":
        if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_pips[0]].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # Left hand
        if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_pips[0]].x:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

# --- Adaptive target selection (Markov / MDP) ---
def transition_prob(current_state, difficulty):
    """Probability distribution over the next target (1-10), shaped by difficulty.

    Easy keeps the next number near the current one; Hard is uniformly random.
    """
    probabilities = np.zeros(10)
    for s in range(1, 11):
        distance = abs(current_state - s)
        if difficulty == 1:  # Easy
            probabilities[s - 1] = 1 / (1 + distance)
        elif difficulty == 2:  # Medium
            probabilities[s - 1] = 1 / (1 + distance**2)
        elif difficulty == 3:  # Hard
            probabilities[s - 1] = 1
    probabilities /= probabilities.sum()
    return probabilities

def mdp_choose_next_target(current_state, difficulty):
    """Sample the next target from the difficulty-shaped distribution."""
    probabilities = transition_prob(current_state, difficulty)
    return int(np.random.choice(range(1, 11), p=probabilities))

# Correct guesses needed to trigger a leaderboard entry, per difficulty
win_threshold = {1: 5, 2: 4, 3: 3}

# --- Game state ---
difficulty_hold_times = {1: 7, 2: 6, 3: 4}
difficulty = 0
target_fingers = random.randint(1, 10)
correct_count = 0
total_correct = 0
tries_left = 0
game_active = False
choosing_difficulty = True
camera_started = False
warm_cold_status = ""
last_finger_count = -1
start_time = 0
grace_start_time = None
flash_color = None
flash_start_time = None
camera_width = 1280
camera_height = 720
difficulty_label = "Difficulty:\n1- Easy\n2- Medium\n3- Hard"
target_history = []
first_try_hint = True

# Leaderboard
leaderboard_list = []
pop_up_active = False
username_input = ""
resumed_once = False

def play_sound(sound_file):
    threading.Thread(target=playsound, args=(sound_file,), daemon=True).start()

cap = cv2.VideoCapture(CAMERA_INDEX)

def draw_buttons(frame):
    cv2.rectangle(frame, (700, 10), (900, 60), (0, 255, 0), -1)
    cv2.putText(frame, "START", (720, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(frame, (950, 10), (1150, 60), (0, 0, 255), -1)
    cv2.putText(frame, "RESTART", (960, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def click_button(x, y):
    if 700 <= x <= 900 and 10 <= y <= 60:
        return "start"
    elif 950 <= x <= 1150 and 10 <= y <= 60:
        return "restart"
    return None

# Pop-up UI coords
popup_x1, popup_y1 = 600, 300
popup_x2, popup_y2 = 1300, 600

def click_popup_button(x, y):
    # Done button
    if (popup_x1+50 <= x <= popup_x1+200) and (popup_y2-70 <= y <= popup_y2-20):
        return "done"
    # Resume button
    if (popup_x1+250 <= x <= popup_x1+400) and (popup_y2-70 <= y <= popup_y2-20):
        return "resume"
    return None

def mouse_callback(event, x, y, flags, param):
    global game_active, choosing_difficulty, camera_started, correct_count, tries_left, target_fingers, warm_cold_status
    global grace_start_time, difficulty_label, target_history, first_try_hint
    global pop_up_active, username_input, total_correct, resumed_once

    if event == cv2.EVENT_LBUTTONDOWN:
        if pop_up_active:
            popup_btn = click_popup_button(x, y)
            if popup_btn == "done":
                if username_input.strip():
                    # Extract difficulty first letter
                    mode_full = difficulty_label.split()[0]
                    mode_map = {"Easy": "E", "Medium": "M", "Hard": "H"}
                    mode_letter = mode_map.get(mode_full, mode_full[0])

                    leaderboard_list.append((username_input.strip(), mode_letter, total_correct))
                    pop_up_active = False
                else:
                    # Ignore if empty username
                    pass

            elif popup_btn == "resume":
                resumed_once = True
                pop_up_active = False
        else:
            button = click_button(x, y)
            if button == "start":
                camera_started = True
            elif button == "restart":
                choosing_difficulty = True
                game_active = False
                camera_started = True  # start again
                correct_count = 0
                total_correct = 0
                tries_left = 0
                target_fingers = random.randint(1, 10)
                target_history = []
                warm_cold_status = ""
                difficulty_label = "Difficulty:\n1- Easy\n2- Medium\n3- Hard"
                grace_start_time = None
                first_try_hint = True
                resumed_once = False
                username_input = ""

cv2.namedWindow("Warm & Cold Game", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Warm & Cold Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Warm & Cold Game", mouse_callback)

def show_popup(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (popup_x1, popup_y1), (popup_x2, popup_y2), (50, 50, 50), -1)
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Instructions
    cv2.putText(frame, "You reached the Target!", (popup_x1+50, popup_y1+50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
    cv2.putText(frame, "Enter Your Name:", (popup_x1+50, popup_y1+120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Text box
    cv2.rectangle(frame, (popup_x1+50, popup_y1+150), (popup_x2-50, popup_y1+220), (255,255,255), 2)
    cv2.putText(frame, username_input, (popup_x1+60, popup_y1+200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Done button
    cv2.rectangle(frame, (popup_x1+50, popup_y2-70), (popup_x1+200, popup_y2-20), (0,255,0), -1)
    cv2.putText(frame, "DONE", (popup_x1+70, popup_y2-35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    # Resume button
    cv2.rectangle(frame, (popup_x1+250, popup_y2-70), (popup_x1+400, popup_y2-20), (0,0,255), -1)
    cv2.putText(frame, "RESUME", (popup_x1+260, popup_y2-35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Handle keyboard input
    key = cv2.waitKey(1)
    if pop_up_active:
        if key != -1:
            if key == 8:  # Backspace
                username_input = username_input[:-1]
            elif 32 <= key <= 126:  # Printable characters
                username_input += chr(key)
    else:
        if key == ord('q'):
            break

    full_screen = cv2.resize(frame, (1920, 1080))
    full_screen.fill(0)

    if camera_started and not pop_up_active:
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        image_height, image_width, _ = frame.shape
        total_fingers = 0

        if results.multi_hand_landmarks:
            # Calculate distances and sort by nearest to camera (z)
            hand_info = []
            for hand_landmarks, hand_class in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = hand_class.classification[0].label
                wrist_z = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z
                hand_info.append((hand_landmarks, hand_label, wrist_z))

            # Sort by wrist_z (closer = smaller z)
            hand_info.sort(key=lambda h: h[2])

            # Process only two nearest hands
            for (hand_landmarks, hand_label, wrist_z) in hand_info[:2]:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                total_fingers += count_fingers(hand_landmarks, hand_label)
                # Bounding box
                x_values = [lm.x for lm in hand_landmarks.landmark]
                y_values = [lm.y for lm in hand_landmarks.landmark]
                min_x = int(min(x_values) * image_width)
                min_y = int(min(y_values) * image_height)
                max_x = int(max(x_values) * image_width)
                max_y = int(max(y_values) * image_height)
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

        # Place camera feed at center
        camera_feed = cv2.resize(frame, (camera_width, camera_height))
        y_offset = (1080 - camera_height) // 2
        x_offset = (1920 - camera_width) // 2
        full_screen[y_offset:y_offset + camera_height, x_offset:x_offset + camera_width] = camera_feed

        # Flash effect
        if flash_color and flash_start_time and time.time() - flash_start_time < 1:
            overlay = full_screen.copy()
            cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + camera_width, y_offset + camera_height), flash_color, -1)
            alpha = 0.2
            full_screen = cv2.addWeighted(overlay, alpha, full_screen, 1 - alpha, 0)
        else:
            flash_color = None

        # Check for steady input
        if total_fingers == last_finger_count:
            if start_time == 0:
                start_time = time.time()
            elif time.time() - start_time >= HOLD_SECONDS:
                if choosing_difficulty:
                    difficulty = total_fingers
                    if difficulty in [1, 2, 3]:
                        tries_left = difficulty_hold_times[difficulty]
                        difficulty_label = ["Easy Difficulty", "Medium Difficulty", "Hard Difficulty"][difficulty - 1]
                        target_fingers = random.randint(1, 10)
                        target_history.append((target_fingers, None))
                        choosing_difficulty = False
                        game_active = True
                        grace_start_time = time.time()
                        start_time = 0
                        correct_count = 0
                        total_correct = 0
                        resumed_once = False
                        username_input = ""
                elif game_active and grace_start_time and time.time() - grace_start_time > GRACE_PERIOD:
                    if total_fingers > 0:
                        if total_fingers == target_fingers:
                            correct_count += 1
                            total_correct += 1
                            warm_cold_status = "Correct!"
                            target_history[-1] = (target_fingers, True)
                            target_fingers = mdp_choose_next_target(total_fingers, difficulty)
                            target_history.append((target_fingers, None))
                            flash_color = (0, 255, 0)
                            flash_start_time = time.time()
                            play_sound(SUCCESS_SOUND)
                            first_try_hint = False

                            # Adjust attempts
                            if difficulty == 1:
                                tries_left += 1
                            elif difficulty == 2:
                                if correct_count % 2 == 0:
                                    tries_left += 1
                            elif difficulty == 3:
                                if correct_count % 2 == 0:
                                    tries_left *= 2

                            # Check threshold
                            threshold = win_threshold[difficulty]
                            if total_correct % threshold == 0:
                                # Reached threshold
                                if resumed_once:
                                    # Immediate save
                                    mode_full = difficulty_label.split()[0]
                                    mode_map = {"Easy": "E", "Medium": "M", "Hard": "H"}
                                    mode_letter = mode_map.get(mode_full, mode_full[0])
                                    name_to_save = username_input.strip() if username_input.strip() else "Unknown"
                                    leaderboard_list.append((name_to_save, mode_letter, total_correct))
                                else:
                                    # Show popup
                                    pop_up_active = True
                        else:
                            tries_left -= 1
                            target_history[-1] = (target_fingers, False)
                            if tries_left > 0:
                                distance = abs(total_fingers - target_fingers)
                                if distance <= 2:
                                    warm_cold_status = "Warm"
                                    flash_color = (0, 0, 255)
                                else:
                                    warm_cold_status = "Cold"
                                    flash_color = (255, 255, 0)
                                flash_start_time = time.time()
                            else:
                                warm_cold_status = "Game Over!"
                                correct_count = 0
                                total_correct = 0
                                play_sound(GAMEOVER_SOUND)
                                choosing_difficulty = True
                                game_active = False
                start_time = 0
        else:
            start_time = 0
            last_finger_count = total_fingers

        # Display Difficulty
        if choosing_difficulty:
            difficulty_lines = difficulty_label.split("\n")
            y_pos = 50
            for line in difficulty_lines:
                cv2.putText(full_screen, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                y_pos += 40
        else:
            cv2.putText(full_screen, difficulty_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Status above camera
        cv2.putText(full_screen, f"Status: {warm_cold_status}", (700, y_offset - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Fingers Count below camera
        cv2.putText(full_screen, f"Your Fingers: {total_fingers}", (700, y_offset + camera_height + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Tries and Score
        cv2.putText(full_screen, f"Tries Left: {tries_left}", (1400, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(full_screen, f"Score: {correct_count}", (1400, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Hint
        if game_active and first_try_hint:
            hint_text = "Hint: Target <= 5" if target_fingers <= 5 else "Hint: Target > 5"
            cv2.putText(full_screen, hint_text, (1400, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 30, 255), 3)

        # Target History on Game Over
        if warm_cold_status == "Game Over!":
            y_pos = 200
            cv2.putText(full_screen, "Target History:", (50, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            y_pos += 40
            for target, correct in target_history:
                color = (255, 255, 255) if correct else (0, 0, 255)
                cv2.putText(full_screen, f"{target}", (50, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                y_pos += 40

        # Display leaderboard on the right side
        leaderboard_x = x_offset + camera_width + 20
        leaderboard_y = 100
        cv2.putText(full_screen, "Leaderboard:", (leaderboard_x, leaderboard_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        leaderboard_y += 50
        # Show last 10 entries
        for uname, mode_letter, score_val in leaderboard_list[-10:]:
            text = f"{uname} - {mode_letter} - {score_val}"
            cv2.putText(full_screen, text, (leaderboard_x, leaderboard_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            leaderboard_y += 40

    if pop_up_active:
        full_screen = show_popup(full_screen)

    # Draw buttons
    draw_buttons(full_screen)

    cv2.imshow("Warm & Cold Game", full_screen)

    # If camera not started and no pop up, quick check for quit
    if not pop_up_active and not camera_started:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
