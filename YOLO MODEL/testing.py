from ultralytics import YOLO
import cv2
import time

# Load the YOLO model
model = YOLO('runs/detect/train12/weights/best.pt')

# Open the camera
cam = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cam.isOpened():
    raise Exception("No Camera")

while True:
    # Capture frame-by-frame
    ret, image = cam.read()
    if not ret:
        break

    # Start time to measure the execution time
    time_mulai = time.time()
    
    # Make predictions on the captured frame
    result = model.predict(image, show=True)
    
    # Print the time taken for prediction
    print("Waktu:", time.time() - time_mulai)
    
    # Display the resulting frame
    _key = cv2.waitKey(1)
    if _key == ord('q'):
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()