#!/usr/bin/env python
from importlib import import_module
import os
from flask import Flask, render_template, Response, send_from_directory
from flask_cors import *
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import threading

# Import the camera driver
from camera_opencv import Camera

app = Flask(__name__)
CORS(app, supports_credentials=True)
camera = Camera()

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

dir_path = os.path.dirname(os.path.realpath(__file__))

@app.route('/api/img/<path:filename>')
def sendimg(filename):
    return send_from_directory(dir_path+'/dist/img', filename)

@app.route('/js/<path:filename>')
def sendjs(filename):
    return send_from_directory(dir_path+'/dist/js', filename)

@app.route('/css/<path:filename>')
def sendcss(filename):
    return send_from_directory(dir_path+'/dist/css', filename)

@app.route('/api/img/icon/<path:filename>')
def sendicon(filename):
    return send_from_directory(dir_path+'/dist/img/icon', filename)

@app.route('/fonts/<path:filename>')
def sendfonts(filename):
    return send_from_directory(dir_path+'/dist/fonts', filename)

@app.route('/<path:filename>')
def sendgen(filename):
    return send_from_directory(dir_path+'/dist', filename)

@app.route('/')
def index():
    return send_from_directory(dir_path+'/dist', 'index.html')

# Initialize global variables for object detection
model = None
interpreter = None
class_labels = ['cup', 'fork', 'glass', 'knife', 'plate', 'spoon']  # Add your class labels here

def initialize_model(model_path, num_threads, enable_edgetpu):
    """Initialize the object detection model."""
    global interpreter
    interpreter = tflite.Interpreter(model_path, num_threads=num_threads)
    if enable_edgetpu:
        interpreter.allocate_tensors()
    else:
        interpreter.allocate_tensors()

def detect_objects(frame):
    """Detect objects in a frame and draw bounding boxes."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    # Preprocess the frame
    frame_resized = cv2.resize(rgb_frame, (input_details['shape'][2], input_details['shape'][1]))
    input_data = frame_resized.reshape(1, *frame_resized.shape).astype(input_details['dtype'])
    interpreter.set_tensor(input_details['index'], input_data)

    # Perform inference
    interpreter.invoke()

    # Get the output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))

    # Draw bounding boxes for detected objects
    for i in range(num_detections):
        class_id = int(classes[0][i])
        class_label = class_labels[class_id]
        confidence = float(scores[0][i])
        if confidence > 0.5:  # Adjust this threshold as needed
            ymin, xmin, ymax, xmax = boxes[0][i]
            im_height, im_width, _ = frame.shape
            left, top, right, bottom = int(xmin * im_width), int(ymin * im_height), int(xmax * im_width), int(ymax * im_height)
            draw_bounding_box(frame, class_label, confidence, left, top, right, bottom)

def draw_bounding_box(frame, label, confidence, x1, y1, x2, y2):
    """Draw a bounding box with label and confidence on the frame."""
    # Draw the bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Create the label text
    label_text = f'{label}: {confidence:.2f}'

    # Calculate text size and position
    (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
    text_x = x1
    text_y = y1 - baseline

    # Draw the label background
    cv2.rectangle(frame, (x1, text_y), (x1 + label_width, text_y - label_height), (0, 0, 255), cv2.FILLED)

    # Put text on the label background
    cv2.putText(frame, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

class webapp:
    def __init__(self):
        self.camera = camera

    def modeselect(self, modeInput):
        Camera.modeSelect = modeInput

    def colorFindSet(self, H, S, V):
        camera.colorFindSet(H, S, V)

    def thread(self):
        app.run(host='0.0.0.0', threaded=True)

    def startthread(self):
        fps_threading=threading.Thread(target=self.thread)         #Define a thread for FPV and OpenCV
        fps_threading.setDaemon(False)                             #'True' means it is a front thread,it would close when the mainloop() closes
        fps_threading.start()                                     #Thread starts

# Function to start object detection
def start_object_detection():
    # Define your TFLite model path and other parameters here
    model_path = 'your_model.tflite'
    num_threads = 4
    enable_edgetpu = False  # Set to True if you're using the Edge TPU

    # Initialize the object detection model
    initialize_model(model_path, num_threads, enable_edgetpu)

    # Start the Flask app for streaming and object detection
    app.run(host='0.0.0.0', threaded=True)

if __name__ == '__main__':
    # Start a thread for object detection
    object_detection_thread = threading.Thread(target=start_object_detection)
    object_detection_thread.start()
