import csv
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import face_recognition
from datetime import datetime
import os
import numpy as np

# Initialize webcam
def initialize_webcam():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    return cap

# Initialize YOLO model for real & fake(img/ video) person
def initialize_yolo_model(model_path):
    return YOLO(model_path)

# Load known face encodings and names from "Photo" folder
def load_known_faces(photo_folder):
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(photo_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            student_img = face_recognition.load_image_file(os.path.join(photo_folder, filename))
            student_encoding = face_recognition.face_encodings(student_img)[0]
            known_face_encodings.append(student_encoding)
            known_face_names.append(os.path.splitext(filename)[0])
    return known_face_encodings, known_face_names

# Initialize CSV file for logging attendance
def initialize_csv_file(attendance_folder):
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    os.makedirs(attendance_folder, exist_ok=True)
    csv_file_path = os.path.join(attendance_folder, current_date + '.csv')
    csv_file = open(csv_file_path, 'w+', newline='')
    csv_writer = csv.writer(csv_file)
    return csv_file, csv_writer

# Process face recognition
def process_faces(frame, boxes, classNames, confidence, known_face_encodings, known_face_names, logged_students, csv_writer, tolerance=0.5):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])

        if conf > confidence:
            if classNames[cls] == 'real':
                small_frame = cv2.resize(frame[y1:y2, x1:x2], (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                        if name not in logged_students:
                            logged_students.add(name)
                            csv_writer.writerow([name, datetime.now().strftime("%H-%M-%S")])

                    face_names.append(name)

                cvzone.cornerRect(frame, (x1, y1, w, h), colorC=(0, 255, 0), colorR=(0, 255, 0))
                if face_names:
                    name_text = ', '.join(face_names)
                    cvzone.putTextRect(frame, f"{name_text.upper()} {int(conf * 100)}%", (max(0, x1), max(35, y1)),
                                       scale=2, thickness=4, colorR=(0, 255, 0), colorB=(0, 255, 0))

            else:
                cvzone.cornerRect(frame, (x1, y1, w, h), colorC=(0, 0, 255), colorR=(0, 0, 255))
                cvzone.putTextRect(frame, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                   (max(0, x1), max(35, y1)),
                                   scale=2, thickness=4, colorR=(0, 0, 255), colorB=(0, 0, 255))

# Main function
def main():
    cap = initialize_webcam()
    model = initialize_yolo_model("models/module_3rd.pt")
    classNames = ["fake", "real"]
    confidence = 0.8
    known_face_encodings, known_face_names = load_known_faces("Photo")
    csv_file, csv_writer = initialize_csv_file("attendance")
    logged_students = set()
    prev_frame_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True, verbose=False)
        for r in results:
            process_faces(frame, r.boxes, classNames, confidence, known_face_encodings, known_face_names, logged_students, csv_writer, tolerance=0.4)

        current_time = time.time()
        fps = 1 / (current_time - prev_frame_time)
        prev_frame_time = current_time
        print(f"FPS: {fps}")

        cv2.imshow('Attendance System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
