
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp

import keras

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_face_mesh = mp.solutions.face_mesh  # Import face_mesh for facial landmarks
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(
        image, 
        results.face_landmarks, 
        mp_face_mesh.FACEMESH_TESSELATION,  # Correct reference
        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    
path ="WIN_20250320_03_51_22_Pro.jpg"

frame = cv2.imread(path)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    frame = cv2.resize(frame, (640, 480)) # Resize image
    image, results = mediapipe_detection(frame, holistic)
# cap = cv2.VideoCapture(0)
# cv2.namedWindow("FullScreen", cv2.WND_PROP_FULLSCREEN)

# cv2.setWindowProperty("FullScreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# # Set mediapipe model 
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():

#         # Read feed
#         ret, frame = cap.read()

#         # Make detections
#         image, results = mediapipe_detection(frame, holistic)
#         # print(results)
        
#         # Draw landmarks
#         draw_styled_landmarks(image, results)


#         # Show to screen
#         cv2.imshow('FullScreen', image)

#         # Break gracefully
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
# draw_landmarks(frame, results)
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))



pose = []
for res in results.pose_landmarks.landmark:
    test = np.array([res.x, res.y, res.z, res.visibility])
    pose.append(test)
pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
result_test = extract_keypoints(results)

np.save('0', result_test)


# # Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data_n') 

# # Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# # Thirty videos worth of data
no_sequences = 90

# # Videos are going to be 30 frames in length
sequence_length = 30

# # Folder start
# start_folder = 0
# # action="thanks"
# # dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
# # for sequence in range(1,start_folder+no_sequences+1):
# #     try: 
# #         os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
# #     except:
# #         pass
    
# # cap = cv2.VideoCapture(0)
# # # Set mediapipe model 
# # with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

# # # NEW LOOP
# # # Loop through actions
# #     action="thanks"
# #     # for action in actions:
# #         # Loop through sequences aka videos
# #     for sequence in range(start_folder, start_folder+no_sequences):
# #         # Loop through video length aka sequence length
# #         for frame_num in range(sequence_length):

# #             # Read feed
# #             ret, frame = cap.read()

# #             # Make detections
# #             image, results = mediapipe_detection(frame, holistic)

# #             # Draw landmarks
# #             draw_styled_landmarks(image, results)
            
# #             # NEW Apply wait logic
# #             if frame_num == 0: 
# #                 cv2.putText(image, 'STARTING COLLECTION', (120,200), 
# #                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
# #                 cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
# #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
# #                 # Show to screen
# #                 cv2.imshow('OpenCV Feed', image)
# #                 cv2.waitKey(500)
# #             else: 
# #                 cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
# #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
# #                 # Show to screen
# #                 cv2.imshow('OpenCV Feed', image)
            
# #             # NEW Export keypoints
# #             keypoints = extract_keypoints(results)
# #             npy_path = os.path.join(DATA_PATH, action, str(sequence+1), str(frame_num))
# #             np.save(npy_path, keypoints)

# #             # Break gracefully
# #             if cv2.waitKey(10) & 0xFF == ord('q'):
# #                 break
                
# # cap.release()
# # cv2.destroyAllWindows()
# # cap.release()
# # cv2.destroyAllWindows()

# from sklearn.model_selection import train_test_split
# # # import tensorflow
# from keras import utils
# label_map = {label:num for num, label in enumerate(actions)}
# label_map
# {'hello': 0, 'thanks': 1, 'iloveyou': 2}
# sequences, labels = [], []
# for action in actions:
#     for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
#         window = []
#         for frame_num in range(sequence_length):
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])
# np.array(sequences).shape

# # np.array(labels).shape

# X = np.array(sequences)
# # X.shape

# y = utils.to_categorical(labels).astype(int)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
# y_test.shape

# # import csv

# # print(X_train)


# # log_dir = os.path.join('Logs')
# # tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
# # model = keras.models.Sequential()
# # model.add(keras.layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
# # model.add(keras.layers.LSTM(128, return_sequences=True, activation='relu'))
# # model.add(keras.layers.LSTM(64, return_sequences=False, activation='relu'))
# # model.add(keras.layers.Dense(64, activation='relu'))
# # model.add(keras.layers.Dense(32, activation='relu'))
# # model.add(keras.layers.Dense(actions.shape[0], activation='softmax'))
# # model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# # model.fit(X_train, y_train, epochs=50, callbacks=[tb_callback])
model=keras.models.load_model("reallylatest2.keras")

# res = model.predict(X_test)
# for i in res:
#     print(actions[np.argmax(res[i])], " : ", actions[np.argmax(y_test[i])])
# print(actions[np.argmax(res[4])])
# 'hello'
# print(actions[np.argmax(y_test[4])])
# 'hello'


# model.save('ASL.keras')
# model.save('reallylatest2.keras')
# del model

# from tensorflow.keras.models import load_model
# from keras.models import load_model()
# Load the trained model
# model = keras.models.load_model('action.h5')
# model.keras.load_weights()

# # Verify model architecture
model.summary()
# model.accuracy(X_test, y_test)

# # model = load_model('action.h5')

# from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
# yhat = model.predict(X_test)
# ytrue = np.argmax(y_test, axis=1).tolist()
# yhat = np.argmax(yhat, axis=1).tolist()
# multilabel_confusion_matrix(ytrue, yhat)

# print(accuracy_score(ytrue, yhat))

from scipy import stats
colors = [(245,117,16), (117,245,16), (16,117,245)] * (len(actions) // 3 + 1)

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    num_actions = min(len(res), len(actions))  

    for num in range(num_actions):
        prob = res[num]
        print(res)
        color = colors[num % len(colors)] 
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), color, -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame

# plt.figure(figsize=(18,18))
# plt.imshow(prob_viz(res, actions, image, colors))


# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
cv2.namedWindow("ASL Detector", cv2.WND_PROP_FULLSCREEN)

cv2.setWindowProperty("ASL Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('ASL Detector', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()