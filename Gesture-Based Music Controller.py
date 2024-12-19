import cv2
import mediapipe as mp
import numpy as np
import pygame

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Initialize Pygame for Track Switching and Volume Control
pygame.init()
pygame.mixer.init()

# Load Tracks
TRACKS = ["boost_bass.wav", "dark_reverb.wav", "delay_effect.wav"]  # Replace with your tracks
current_track_index = 0
pygame.mixer.music.set_volume(0.5)  # Set initial volume

# Function to load and play a track
def play_track(index):
    global current_track_index
    current_track_index = index % len(TRACKS)
    pygame.mixer.music.load(TRACKS[current_track_index])
    pygame.mixer.music.play(-1)

# Start with the first track
play_track(current_track_index)

# Function to calculate Euclidean distance
def calculate_distance(point1, point2, frame_width, frame_height):
    x1, y1 = int(point1.x * frame_width), int(point1.y * frame_height)
    x2, y2 = int(point2.x * frame_width), int(point2.y * frame_height)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), (x1, y1), (x2, y2)

# Function to count raised fingers
def count_raised_fingers(hand_landmarks, frame_width, frame_height):
    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    finger_pips = [
        mp_hands.HandLandmark.THUMB_IP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP,
    ]

    count = 0
    for tip, pip in zip(finger_tips, finger_pips):
        tip_y = hand_landmarks.landmark[tip].y * frame_height
        pip_y = hand_landmarks.landmark[pip].y * frame_height
        if tip_y < pip_y:  # If tip is above pip, the finger is raised
            count += 1
    return count

# Function to detect gestures
def detect_gesture(finger_count, thumb_index_distance):
    if finger_count == 0:
        return "Fist"
    elif finger_count == 5:
        return "Open Hand"
    elif finger_count == 1 and thumb_index_distance < 50:
        return "Pinch"
    elif finger_count == 2:
        return "Peace"
    else:
        return "Unknown"

# Function to calculate and display distances between specified finger pairs
def calculate_finger_distances(hand_landmarks, frame_width, frame_height):
    # Specify pairs of fingers to calculate distances
    finger_pairs = [
        (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP),
        (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP),
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP),
    ]

    distances = []
    for pair in finger_pairs:
        point1 = hand_landmarks.landmark[pair[0]]
        point2 = hand_landmarks.landmark[pair[1]]
        distance, coord1, coord2 = calculate_distance(point1, point2, frame_width, frame_height)
        distances.append((distance, coord1, coord2))

    return distances

# Start capturing video
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

previous_hand_positions = {}
previous_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and process the frame
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    hand_positions = []
    current_time = cv2.getTickCount()
    if previous_time is not None:
        time_elapsed = (current_time - previous_time) / cv2.getTickFrequency()
    else:
        time_elapsed = 0
    previous_time = current_time

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            # Draw the hand annotations
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Identify hand type
            hand_type = hand_handedness.classification[0].label
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            current_position = (int(wrist.x * frame_width), int(wrist.y * frame_height))

            if hand_type not in previous_hand_positions:
                previous_hand_positions[hand_type] = None

            # Count raised fingers and calculate gesture
            finger_count = count_raised_fingers(hand_landmarks, frame_width, frame_height)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_index_distance, thumb_coords, index_coords = calculate_distance(
                thumb_tip, index_tip, frame_width, frame_height
            )
            gesture = detect_gesture(finger_count, thumb_index_distance)

            # Perform gesture-based actions
            if gesture == "Pinch":
                current_track_index += 1
                play_track(current_track_index)
            elif gesture == "Peace":
                pygame.mixer.music.pause()
            elif gesture == "Open Hand":
                pygame.mixer.music.unpause()

            # Display hand-specific information
            if hand_type == "Right":
                x_pos = frame_width - 300
                text_color = (0, 255, 0)
            else:
                x_pos = 50
                text_color = (255, 0, 0)

            cv2.putText(frame, f"{hand_type} Hand", (x_pos, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            cv2.putText(frame, f"Fingers: {finger_count}", (x_pos, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            cv2.putText(frame, f"Gesture: {gesture}", (x_pos, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

            hand_positions.append(current_position)

            # Calculate and display finger-to-finger distances
            finger_distances = calculate_finger_distances(hand_landmarks, frame_width, frame_height)
            for distance, coord1, coord2 in finger_distances:
                # Draw lines between the points
                cv2.line(frame, coord1, coord2, (255, 0, 255), 2)
                # Display the distance
                middle_x = (coord1[0] + coord2[0]) // 2
                middle_y = (coord1[1] + coord2[1]) // 2
                cv2.putText(frame, f"{distance:.1f}px", (middle_x, middle_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Calculate and display distance between two hands
        if len(hand_positions) == 2:
            middle_x = (hand_positions[0][0] + hand_positions[1][0]) // 2
            middle_y = (hand_positions[0][1] + hand_positions[1][1]) // 2
            distance = np.linalg.norm(np.array(hand_positions[0]) - np.array(hand_positions[1]))
            cv2.line(frame, hand_positions[0], hand_positions[1], (255, 0, 0), 2)
            cv2.putText(frame, f"Distance: {distance:.2f}px", (middle_x - 100, middle_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow("Gesture-Based Music Controller with Finger and Hand Distance", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
pygame.mixer.music.stop()
pygame.quit()
