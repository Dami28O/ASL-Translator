import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np

def main():

    # load in the model
    model = tf.keras.models.load_model("SLRecogniser.h5")

    # Initialize MediaPipe for hand detection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    # initialise mp for drawing
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    
    # get footage from webcam
    cap = cv2.VideoCapture(0)

    # Set FPS to 15 - so don't have to read every frame and can reduce computational load
    cap.set(cv2.CAP_PROP_FPS, 15) 

    # a list of symbols over each frame
    frames = []
    string = []
    over = 0

    # initialisation of hands solution
    with mp_hands.Hands(
    model_complexity=0, # less accuracy but more speed
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
        
        # loop over feed
        while cap.isOpened():

            # read every frame
            success, frame = cap.read()

            # check that the frames are read successfully
            if not success:
                print("Frames not read correctly")
                continue

            # process the frame
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Flip the frame horizontally to create a mirror
            frame = cv2.flip(frame, 1)
            # flip image so it lines up with the frame
            image_rgb = cv2.flip(image_rgb, 1)
            
            # perform the detection of landmarks
            detection_result = hands.process(image_rgb)

            # to print text
            text = False

            # if we detect something
            if detection_result.multi_hand_landmarks:

                # list of the landmarks
                landmarks = []
                flattened_landmarks = []

                # go over every single landmark in in the hand and save to the landmarks list (loop for if we were doing two hands)
                for hand_landmarks in detection_result.multi_hand_landmarks:
                    
                    # draw over the image with the landmarks 
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    for landmark in hand_landmarks.landmark:
                        # access individual x, y, z coordinates
                        landmarks.append([landmark.x, landmark.y, landmark.z])

                # flatten into a 1D array for the NN
                flattened_landmarks = [item for sublist in landmarks for item in sublist]

                # if two hands detected?
                if len(flattened_landmarks) > 63:
                    continue

                # change to a numpy array to be fed into the model
                data = np.array(flattened_landmarks).reshape(1,63)

                # pass into the model to get an output
                prediction = model.predict(data)

                # Get the related label
                prediction_label = np.argmax(prediction)

                # decode the result label
                result = decode(prediction_label)

                # print text
                text = True

                # add the letter to frames
                frames.append(result)
            
            if text:
                # print the label
                cv2.putText(frame, f"Letter: {result}", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2, lineType=cv2.LINE_AA)
            # check if we have 30 of frames of the same symbol
            if len(frames) > 30 and len(set(frames)) == 1:
                if frames[0] == 'space':
                    string.append(' ')
                elif frames[0] == 'del':
                    string.pop(-1)
                else:
                    string.append(frames[0])
                    
                frames.clear()
            elif len(frames) > 30:
                # remove the oldest elements to keep frames at a fixed length of 30
                over = len(frames) - 30
                for i in range(over):
                    frames.pop(0)

            sentence = ''.join(string)

            cv2.putText(frame, str(sentence), (10, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, lineType=cv2.LINE_AA)

            # flip show the image
            cv2.imshow('MediaPipe Hands', frame)
        
            # exit the loop if the user presses escape
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


def decode(prediction):

    if prediction == 26:
        result = 'space'
    elif prediction == 27:
        result = 'del'
    else:
        # return to ascii
        prediction += 65
        result = chr(prediction)
    
    return result


if __name__ == "__main__":
    main()