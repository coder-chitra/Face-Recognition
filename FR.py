import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime

# Path to images folder
path = 'Images'
images = []
classNames = []
imageList = os.listdir(path)

# Load images and extract names
for img_name in imageList:
    curImg = cv2.imread(f'{path}/{img_name}')
    curImg = cv2.resize(curImg, (300, 300))  # Resize for faster processing
    images.append(curImg)
    classNames.append(os.path.splitext(img_name)[0])  # Extract name without extension

# Function to encode known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        encode = face_recognition.face_encodings(img)
        if encode:  # Check if encoding exists
            encodeList.append(encode[0])
        else:
            print(f"Warning: No face detected in image")
    return encodeList

encodeListKnown = findEncodings(images)
print("✅ Encoding Complete")

# Dictionary to track login/logout times
attendance_dict = {}

# Function to mark login time
def markLogin(name):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if name not in attendance_dict:
        attendance_dict[name] = {'Login Time': now, 'Logout Time': None}

# Function to update logout time
def markLogout(name):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if name in attendance_dict and attendance_dict[name]['Logout Time'] is None:
        attendance_dict[name]['Logout Time'] = now

# Open webcam
cap = cv2.VideoCapture(0)
frame_count = 0
last_seen_name = None
last_face_location = None
last_detected_time = None

while True:
    success, img = cap.read()
    frame_count += 1

    if frame_count % 5 == 0:  # Process every 5th frame
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Reduce size for faster processing
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        detected_name = None  # Reset detected name
        detected_face_location = None  # Reset detected face location

        if len(encodesCurFrame) > 0:
            last_detected_time = None  # Reset logout timer
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    detected_name = classNames[matchIndex].upper()
                    detected_face_location = faceLoc
                    markLogin(detected_name)

        # If face disappears, start logout countdown
        if last_seen_name and detected_name is None:
            if last_detected_time is None:
                last_detected_time = datetime.now()

            time_diff = (datetime.now() - last_detected_time).total_seconds()
            if time_diff > 5:  # If absent for 5+ seconds, mark logout
                markLogout(last_seen_name)
                last_seen_name = None
                last_face_location = None  # Clear rectangle immediately

        last_seen_name = detected_name  # Update last seen person
        last_face_location = detected_face_location  # Update face location

    # Draw rectangle and name **ONLY if a face is detected**
    if last_seen_name and last_face_location:
        y1, x2, y2, x1 = last_face_location
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale back up

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, last_seen_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Ensure everyone logged out when program stops
for name in list(attendance_dict.keys()):
    if attendance_dict[name]['Logout Time'] is None:
        markLogout(name)

cap.release()
cv2.destroyAllWindows()

# Save attendance to CSV
df = pd.DataFrame.from_dict(attendance_dict, orient='index')
df.reset_index(inplace=True)
df.columns = ['Name', 'Login Time', 'Logout Time']
df.to_csv('attendance.csv', index=False)

print("✅ Attendance saved to attendance.csv")
