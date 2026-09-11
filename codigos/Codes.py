import cv2  
import time  
import mediapipe as mp 
import numpy as np  
import openai  
import speech_recognition as sr 

def handTracking(show, number, changeColor, color, question):
    """
    This function performs hand tracking using MediaPipe's Hand module and provides various functionalities:
    
    - show: If True, it displays the hand landmarks on the camera feed.
    - number: If True, it adds numbers to the landmarks for identification.
    - changeColor: A list of landmark indices whose color will be changed in the displayed hand landmarks.
    - color: The color to change the specified landmarks to.
    - question: If True, it recognizes speech input when a hand covers a certain area and generates text responses.

    Make sure to set your OpenAI API key before using this function.
    """
    # Set your OpenAI API key here
    openai.api_key = "APIKEY"

    # Initialize the speech recognizer
    recognizer = sr.Recognizer()

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Initialize MediaPipe Drawing
    mp_drawing = mp.solutions.drawing_utils

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            if show:
                for landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                    if question:
                        # Get the coordinates of the bounding box
                        x_min = min(landmark.x for landmark in landmarks.landmark)
                        y_min = min(landmark.y for landmark in landmarks.landmark)
                        x_max = max(landmark.x for landmark in landmarks.landmark)
                        y_max = max(landmark.y for landmark in landmarks.landmark)

                        # Draw a rectangle around the hand
                        cv2.rectangle(frame, (int(x_min * frame.shape[1]), int(y_min * frame.shape[0])),
                                      (int(x_max * frame.shape[1]), int(y_max * frame.shape[0])),
                                      (0, 255, 0), 2)

                        # Calculate the area of the rectangle containing the hand
                        area = (x_max - x_min) * (y_max - y_min) * 1000

                        # If the area of the rectangle is greater than a threshold, recognize speech
                        if area > 170:  # Check if the area of the hand's bounding box is larger than a threshold
                            with sr.Microphone() as source:
                                print("Speak now...")  # Prompt the user to speak
                                recognizer.adjust_for_ambient_noise(source)  # Adjust microphone for ambient noise
                                audio = recognizer.listen(source)  # Listen to audio input
                                print("Processing...")  # Indicate that audio processing is in progress
                            try:
                                question = recognizer.recognize_google(audio, language="es-ES")  # Recognize speech using Google Web API
                            except sr.UnknownValueError:
                                print("Could not understand audio")  # Handle unknown speech
                            except sr.RequestError as e:
                                print("Request error: {0}".format(e))  # Handle API request error
    
                            if question:
                                # Generate a response to the recognized question using OpenAI's text generation
                                response = openai.Completion.create(
                                    engine="text-davinci-002",
                                    prompt=question,
                                    max_tokens=50  # Adjust the number of tokens in the response as needed
                                )
                                print("Question:", question)  # Print the recognized question
                                print("Response:", response.choices[0].text)  # Print the generated response
                                print('\nGreat! Now, move your hand away.')  # Prompt the user to move their hand away
    
                            time.sleep(5)  # Pause for 5 seconds to avoid continuous processing
    
                    if number:
                        # Add numbers to landmarks for identification
                        for idx, landmark in enumerate(landmarks.landmark):
                            h, w, _ = frame.shape
                            cx, cy = int(landmark.x * w), int(landmark.y * h)
                            cv2.putText(frame, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
                    if changeColor is not None:
                        for mark in changeColor:
                            # Change the color of specific landmarks
                            landmark_index_to_color = mark
                            landmark_to_change = landmarks.landmark[landmark_index_to_color]
                            cx, cy = int(landmark_to_change.x * frame.shape[1]), int(landmark_to_change.y * frame.shape[0])
                            cv2.circle(frame, (cx, cy), 5, color, -1)  # Change color to red (BGR format)
    
        if question:
            cv2.putText(frame, 'Bring your hand closer to ask a question to chatGPT',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Hand Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit the loop when 'q' key is pressed
    
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
