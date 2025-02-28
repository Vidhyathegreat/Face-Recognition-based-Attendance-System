import cv2
from flask import Flask, render_template, Response, request, jsonify
import mysql.connector
import os
import time
import numpy as np
import logging

# Set up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# MySQL database connection
try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Vidhya@12.",  # Use your MySQL password here
        database="attendance_db"
    )
    cursor = db.cursor()
    logging.info("Connected to the MySQL database.")
except mysql.connector.Error as err:
    logging.error(f"Error connecting to the database: {str(err)}")
    raise

# Load the pre-trained face detector model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the recognizer for face recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to train the recognizer with stored images
def train_face_recognizer():
    folder_path = r"C:\Projects\facerecognition\images"
    face_samples = []
    student_ids = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                logging.warning(f"Failed to load image: {image_path}")
                continue

            face_samples.append(image)

            try:
                student_id = int(os.path.splitext(filename)[0])
                student_ids.append(student_id)
            except ValueError:
                logging.warning(f"Filename does not contain a valid student ID: {filename}")
                continue

    if face_samples:
        recognizer.train(face_samples, np.array(student_ids))
        logging.info("Face recognizer trained successfully.")
    else:
        logging.error("No valid images or IDs found. Training failed.")
        raise Exception("No valid images or IDs found.")

# Train the face recognizer
try:
    train_face_recognizer()
except Exception as e:
    logging.error(f"Error during training: {str(e)}")

# Render the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Function to generate video frames from the webcam
def generate_video_frames():
    camera = cv2.VideoCapture(0)  # Capture from the default webcam
    while True:
        is_frame_captured, frame = camera.read()
        if not is_frame_captured:
            logging.error("Failed to capture video frame.")
            break
        else:
            # Encode the frame as JPEG
            ret, jpeg_buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame = jpeg_buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                logging.error("Failed to encode frame.")

# Route to stream video feed to the client
@app.route('/video_feed')
def video_feed():
    return Response(generate_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to save student details to the database
@app.route('/save_details', methods=['POST'])
def save_student_details():
    try:
        # Get form data
        student_name = request.form['name']
        student_id = request.form['student_id']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        branch = request.form['branch']

        # Insert or update student details in the database
        sql = """INSERT INTO student_info (student_id, student_name, email, phone, address, branch)
                 VALUES (%s, %s, %s, %s, %s, %s)
                 ON DUPLICATE KEY UPDATE student_name=%s, email=%s, phone=%s, address=%s, branch=%s"""
        cursor.execute(sql, (student_id, student_name, email, phone, address, branch, student_name, email, phone, address, branch))
        db.commit()
        logging.info(f"Student details saved for ID: {student_id}")
        return jsonify({'status': 'success', 'message': 'Student details saved.'})

    except Exception as e:
        logging.error(f"Error saving student details: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Failed to save student details.'})

# Route to capture and recognize a student's face
@app.route('/capture_image', methods=['POST'])
def capture_student_image():
    try:
        student_id = request.form['student_id']

        # Capture image from webcam
        camera = cv2.VideoCapture(0)
        is_frame_captured, frame = camera.read()

        if not is_frame_captured:
            logging.error("Failed to capture image.")
            return jsonify({'status': 'error', 'message': 'Failed to capture image.'})

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))

        if len(detected_faces) > 0:
            matched_image_path = compare_faces(gray_frame, detected_faces, student_id)
            if matched_image_path:
                mark_student_attendance(student_id)
                return jsonify({'status': 'success', 'message': 'Attendance marked.'})
            else:
                return jsonify({'status': 'fail', 'message': 'No matching face found.'})
        else:
            logging.warning("No face detected in the image.")
            return jsonify({'status': 'fail', 'message': 'No face detected.'})

    except Exception as e:
        logging.error(f"Error during image capture or recognition: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Error during image capture or recognition.'})

# Compare captured face with stored images
def compare_faces(gray_frame, detected_faces, student_id):
    folder_path = r"C:\Projects\facerecognition\images"
    match_threshold = 70  # Lower value for stricter matching

    for (x, y, w, h) in detected_faces:
        captured_face = gray_frame[y:y+h, x:x+w]
        resized_captured_face = cv2.resize(captured_face, (200, 200))

        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                stored_image_path = os.path.join(folder_path, filename)
                stored_image = cv2.imread(stored_image_path, cv2.IMREAD_GRAYSCALE)
                resized_stored_face = cv2.resize(stored_image, (200, 200))
                label, confidence = recognizer.predict(resized_stored_face)

                if confidence < match_threshold and filename.startswith(student_id):
                    logging.info(f"Face matched for student ID: {student_id} with confidence: {confidence}")
                    return stored_image_path

    logging.info(f"No matching face found for student ID: {student_id}")
    return None

# Mark student attendance
def mark_student_attendance(student_id):
    current_date = time.strftime('%Y-%m-%d')

    # Check if attendance already marked for today
    cursor.execute("SELECT * FROM attendance WHERE student_id=%s AND DATE(timestamp)=%s", (student_id, current_date))
    attendance_record = cursor.fetchone()

    if attendance_record is None:
        # Mark attendance
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        sql = """INSERT INTO attendance (student_id, timestamp) VALUES (%s, %s)"""
        cursor.execute(sql, (student_id, timestamp))
        db.commit()
        logging.info(f"Attendance marked for student ID: {student_id}")
    else:
        logging.info(f"Attendance already marked for student ID: {student_id} today.")

# Run the Flask app with proper host and port configuration
if __name__ == "__main__":
    try:
        logging.info("Starting the Flask application...")
        # Bind to all IP addresses ('0.0.0.0') and specify port (5000 by default)
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except Exception as e:
        logging.error(f"Error running the application: {str(e)}")
