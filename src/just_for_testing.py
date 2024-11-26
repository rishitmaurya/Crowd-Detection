import cv2
import os
import torch  # Ensure torch is installed

# Function to perform crowd detection and display results
def detect_crowd(source, model_path="models/yolov5x-seg.pt", conf_threshold=0.3):
    # Load YOLOv5 model for crowd detection
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # Load your custom trained model if you have one

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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 files

    # Determine the output file name and format
    filename, ext = os.path.splitext(os.path.basename(source))
    output_video_path = os.path.join(output_folder, f"{filename}_crowd_detected{ext}")

    # Set up VideoWriter to save the output video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame (BGR format) to RGB and then to a tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)  # Change shape to (C, H, W)
        frame_tensor = frame_tensor.float() / 255.0  # Normalize to [0, 1]
        frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension

        # Run YOLOv5 detection on the frame
        results = model(frame_tensor)

        # Process the results from the segmentation model
        if isinstance(results, torch.Tensor):
            masks = results['masks']  # Get masks if using segmentation model
            boxes = results['boxes']  # Get boxes
            scores = results['scores']  # Get scores
            
            # Filter results by confidence threshold
            keep = scores > conf_threshold
            masks = masks[keep]
            boxes = boxes[keep]

            crowd_count = keep.sum().item()  # Count the number of detections

            # Draw bounding boxes and masks
            for i in range(len(boxes)):
                box = boxes[i]
                mask = masks[i]

                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.tolist())  # Convert to integer

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Optionally, you can also draw the mask if needed
                # Here you would apply the mask to the frame if required

            # Display crowd count on frame
            cv2.putText(frame, f"Crowd Count: {crowd_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Write the processed frame to the output video
        out.write(frame)

        # Resize the frame for display (adjust size as needed)
        display_width = int(width * 0.5)  # 50% of original width
        display_height = int(height * 0.5)  # 50% of original height
        resized_frame = cv2.resize(frame, (display_width, display_height))

        # Show the resized frame with bounding boxes
        cv2.imshow("Crowd Detection", resized_frame)

        # Break loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()  # Release the VideoWriter
    cv2.destroyAllWindows()

# Example usage for testing
detect_crowd("test-dataset/crowd-mess2.mp4", model_path="models/yolov5x-seg.pt", conf_threshold=0.3)
