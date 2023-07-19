import cv2
from flask import Flask, render_template, Response
import numpy as np
import atexit

app = Flask(__name__)

# Load the pre-trained YOLO model and class labels
net = cv2.dnn.readNetFromDarknet('F:\Projects PiZone\python projects\Person Detection\yolo\yolov4.cfg', 'F:\Projects PiZone\python projects\Person Detection\yolo\yolov4.weights')
classes = []
with open('F:\Projects PiZone\python projects\Person Detection\yolo\classes.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

camera = None

def initialize_camera():
    global camera
    camera = cv2.VideoCapture0("F:\Projects PiZone\python projects\Person Detection\video_for_count.mp4")

def release_camera():
    global camera
    if camera is not None:
        camera.release()

def generate_frames():
    while True:
         # Read frame from the camera
        ret, frame = camera.read()
        if not ret:
            break

         # Prepare the frame for object detection
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

          # Perform forward pass through the network
        output_layers = net.getUnconnectedOutLayersNames()
        outputs = net.forward(output_layers)

        # Initialize lists to store detected person bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

         # Process detections
        for output in outputs:
            for detection in output:
                scores = np.array(detection[5:])
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and classes[class_id] == "person":
                     # Get bounding box coordinates
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (x, y, w, h) = box.astype("int")
                    
                     # Calculate the top-left corner coordinates of the bounding box
                    x = int(x - (w / 2))
                    y = int(y - (h / 2))

                     # Append the person detection to the lists   
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression to eliminate redundant overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        person_count = 0

         # Draw bounding boxes and count the number of persons
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = f"Person: {confidences[i]:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                person_count += 1

        # Display the frame with bounding boxes and person count
        cv2.putText(frame, f"Person Count: {person_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Convert the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

         # Yield the frame in byte stream format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('person.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop', methods=['POST'])
def stop():
    release_camera()
    return render_template('stopped.html')

if __name__ == '__main__':
    initialize_camera()
    atexit.register(release_camera)
    app.run(debug=True)
