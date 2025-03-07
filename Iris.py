import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import face_recognition
from skimage.filters import gabor
from scipy.spatial.distance import hamming

# Path to images folder
path = 'Images'
images = []
iris_images = []
classNames = []
imageList = os.listdir(path)

# Load images and extract names
for img_name in imageList:
    curImg = cv2.imread(f'{path}/{img_name}')
    grayImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY)
    curImg = cv2.resize(curImg, (500, 500))  # Increased resolution
    images.append(curImg)
    iris_images.append(cv2.resize(grayImg, (150, 150)))
    classNames.append(os.path.splitext(img_name)[0])

# Function to encode known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
        else:
            print("Warning: No face detected in image")
    return encodeList

# Function to extract iris features using Gabor filters
def extract_iris_features(image):
    image = cv2.equalizeHist(image)  # Enhance contrast
    filtered, _ = gabor(image, frequency=0.2)
    return filtered.flatten()

encodeListKnown = findEncodings(images)
iris_encodeListKnown = [extract_iris_features(img) for img in iris_images]
print("✅ Encoding Complete")

# Dictionary to track login/logout times
attendance_dict = {}

def markLogin(name):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if name not in attendance_dict:
        attendance_dict[name] = {'Login Time': now, 'Logout Time': None}
        print(f"✅ {name} logged in at {now}")

def markLogout(name):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if name in attendance_dict and attendance_dict[name]['Logout Time'] is None:
        attendance_dict[name]['Logout Time'] = now
        print(f"❌ {name} logged out at {now}")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

frame_count = 0
last_seen_name = None
last_detected_time = None

while True:
    success, img = cap.read()
    frame_count += 1
    detected_name = None
    face_rectangle = None

    if frame_count % 5 == 0:  # Process every 5th frame
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex] and faceDis[matchIndex] < 0.5:
                detected_name = classNames[matchIndex].upper()
                markLogin(detected_name)
                y1, x2, y2, x1 = faceLoc
                face_rectangle = (x1 * 4, y1 * 4, x2 * 4, y2 * 4)

        # Eye/Iris Detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in eyes:
            iris_roi = gray[y:y+h, x:x+w]
            iris_roi = cv2.resize(iris_roi, (150, 150))
            iris_features = extract_iris_features(iris_roi)
            distances = [hamming(iris_features, known) for known in iris_encodeListKnown]
            matchIndex = np.argmin(distances)
            if distances[matchIndex] < 0.25:
                detected_name = classNames[matchIndex].upper()
                markLogin(detected_name)
                face_rectangle = (x, y, x + w, y + h)

        # If no face or iris detected, check for logout
        if last_seen_name and detected_name is None:
            if last_detected_time is None:
                last_detected_time = datetime.now()
            if (datetime.now() - last_detected_time).total_seconds() > 5:
                markLogout(last_seen_name)
                last_seen_name = None
                face_rectangle = None

        last_seen_name = detected_name

    # Draw rectangle and name
    if face_rectangle and last_seen_name:
        x1, y1, x2, y2 = face_rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, last_seen_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Face & Iris Recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ensure all users are logged out
for name in list(attendance_dict.keys()):
    if attendance_dict[name]['Logout Time'] is None:
        markLogout(name)

cap.release()
cv2.destroyAllWindows()

# Save attendance
df = pd.DataFrame.from_dict(attendance_dict, orient='index')
df.reset_index(inplace=True)
df.columns = ['Name', 'Login Time', 'Logout Time']
df.to_csv('face_iris_attendance.csv', index=False)
print("✅ Attendance saved to face_iris_attendance.csv")


