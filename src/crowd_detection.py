import cv2
import os
from ultralytics import YOLO

# Function to perform detection and display results with improvements
def detect_people(source, model_path="models/yolov8x.pt", conf_threshold=0.1, input_size=1280):
    # Load YOLOv8 model with specified input size and confidence threshold
    model = YOLO(model_path)
    model.conf = conf_threshold  # Set confidence threshold

    # Create output folder for saving detected frames
    output_folder = "detected"
    os.makedirs(output_folder, exist_ok=True)

    # Open the video source
    cap = cv2.VideoCapture(source)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video

    # Determine the output file name and format
    filename, ext = os.path.splitext(os.path.basename(source))
    output_video_path = os.path.join(output_folder, f"{filename}_detected{ext}")

    # Set up VideoWriter to save the output video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0  # Frame counter
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocessing: Improve contrast/brightness if needed
        # Adjust contrast (alpha) and brightness (beta) values as needed
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

        # Run YOLOv8 detection on the frame with specified input size
        results = model(frame, imgsz=input_size)  # Set custom input size for better detection of smaller objects

        people_count = 0  # Initialize people count
        for result in results[0].boxes:
            if result.cls == 0:  # '0' corresponds to 'person' class
                people_count += 1
                x1, y1, x2, y2 = result.xyxy[0].int().tolist()  # Get bounding box coordinates
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display people count on frame
        # cv2.putText(frame, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Define the text and background position
        text = f"People Count: {people_count}"
        position = (10, 50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        text_color = (0, 0, 255)  # Red color for the text
        bg_color = (255, 255, 255)  # White color for the background

        # Get the width and height of the text box
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Draw the background rectangle with padding
        cv2.rectangle(frame, (position[0], position[1] - text_height - baseline),
                    (position[0] + text_width, position[1] + baseline),
                    bg_color, thickness=cv2.FILLED)

        # Put the text on top of the rectangle
        cv2.putText(frame, text, position, font, font_scale, text_color, font_thickness)
        
        # Write the processed frame to the output video
        out.write(frame)

        # Resize the frame for display (adjust size as needed)
        display_width = int(width * 0.5)  # 50% of original width
        display_height = int(height * 0.5)  # 50% of original height
        resized_frame = cv2.resize(frame, (display_width, display_height))

        # Show the resized frame with bounding boxes
        cv2.imshow("People Detection", resized_frame)

        # Break loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1  # Increment the frame counter

    # Release resources
    cap.release()
    out.release()  # Release the VideoWriter
    cv2.destroyAllWindows()

# Example usage for testing
# Adjust model path and source video path as needed
# detect_people("test-dataset/crowd-mess2.mp4", model_path="models/yolov8x.pt", conf_threshold=0.1, input_size=1280)