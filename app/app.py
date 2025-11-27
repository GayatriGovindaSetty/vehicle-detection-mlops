import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import VehicleDetector
from src.config import config
import os

# Check if model exists, use default if not
model_path = "models/best.pt" if os.path.exists("models/best.pt") else "yolov8n.pt"

# Initialize detector
try:
    detector = VehicleDetector(model_path=model_path)
except Exception as e:
    print(f"Warning: Could not load model. {e}")
    detector = None

def detect_vehicles(image, conf_threshold, iou_threshold):
    """
    Detect vehicles in the uploaded image
    
    Args:
        image: Input image from Gradio
        conf_threshold: Confidence threshold
        iou_threshold: IOU threshold
        
    Returns:
        Annotated image and detection statistics
    """
    if image is None:
        return None, "Please upload an image"
    
    if detector is None:
        return None, "⚠️ Model not loaded. Please train and add best.pt to models/ folder"
    
    # Update thresholds
    detector.conf_threshold = conf_threshold
    detector.iou_threshold = iou_threshold
    
    try:
        # Run detection
        result, annotated_image = detector.detect(image, return_annotated=True)
    
    # Get statistics
    stats = detector.get_detection_stats(result)
    
    # Format statistics for display
    stats_text = f"""
    ### Detection Results
    
    **Total Detections:** {stats['total_detections']}
    
    **Detected Vehicles by Class:**
    """
    
    for class_name, count in stats['class_counts'].items():
        stats_text += f"\n- {class_name}: {count}"
    
    stats_text += f"\n\n**Average Confidence:** {stats['average_confidence']:.2%}"
    
    if stats['total_detections'] > 0:
        stats_text += "\n\n**Individual Detections:**\n"
        for i, det in enumerate(stats['detections'], 1):
            stats_text += f"\n{i}. {det['class']} (confidence: {det['confidence']:.2%})"
    
    return annotated_image, stats_text
    """
    Detect vehicles in video (process sample frames)
    
    Args:
        video_path: Path to video file
        conf_threshold: Confidence threshold
        iou_threshold: IOU threshold
        skip_frames: Process every nth frame
        
    Returns:
        Processed video and statistics
    """
    if video_path is None:
        return None, "Please upload a video"
    
    # Update thresholds
    detector.conf_threshold = conf_threshold
    detector.iou_threshold = iou_threshold
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare output video
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    all_stats = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every nth frame
        if frame_count % skip_frames == 0:
            result, annotated = detector.detect(frame, return_annotated=True)
            stats = detector.get_detection_stats(result)
            total_detections += stats['total_detections']
            all_stats.append(stats)
            
            # Convert RGB back to BGR for video writer
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            out.write(annotated_bgr)
        else:
            out.write(frame)
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Calculate summary statistics
    avg_detections = total_detections / len(all_stats) if all_stats else 0
    
    summary = f"""
    ### Video Processing Results
    
    **Total Frames Processed:** {len(all_stats)}
    **Total Detections:** {total_detections}
    **Average Detections per Frame:** {avg_detections:.2f}
    """
    
    return output_path, summary

# Create Gradio interface
with gr.Blocks(title="Vehicle Detection System", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🚗 Real-Time Vehicle Detection System
        
        This system uses YOLOv8 to detect 8 classes of vehicles:
        **Auto, Bus, Car, LCV, Motorcycle, Multiaxle, Tractor, Truck**
        
        Upload an image or video to detect vehicles!
        """
    )
    
    with gr.Tab("Image Detection"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image", type="numpy")
                
                with gr.Row():
                    conf_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                        label="Confidence Threshold"
                    )
                    iou_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.45, step=0.05,
                        label="IOU Threshold"
                    )
                
                detect_btn = gr.Button("Detect Vehicles", variant="primary")
                
                gr.Examples(
                    examples=[],  # Add example images here
                    inputs=image_input,
                    label="Example Images"
                )
            
            with gr.Column():
                image_output = gr.Image(label="Detection Result")
                stats_output = gr.Markdown(label="Statistics")
        
        detect_btn.click(
            fn=detect_vehicles,
            inputs=[image_input, conf_slider, iou_slider],
            outputs=[image_output, stats_output]
        )
    
    with gr.Tab("Video Detection"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                
                with gr.Row():
                    conf_slider_vid = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                        label="Confidence Threshold"
                    )
                    iou_slider_vid = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.45, step=0.05,
                        label="IOU Threshold"
                    )
                
                skip_frames_slider = gr.Slider(
                    minimum=1, maximum=10, value=2, step=1,
                    label="Process Every Nth Frame (for speed)"
                )
                
                process_btn = gr.Button("Process Video", variant="primary")
            
            with gr.Column():
                video_output = gr.Video(label="Processed Video")
                video_stats = gr.Markdown(label="Statistics")
        
        process_btn.click(
            fn=detect_from_video,
            inputs=[video_input, conf_slider_vid, iou_slider_vid, skip_frames_slider],
            outputs=[video_output, video_stats]
        )
    
    with gr.Tab("About"):
        gr.Markdown(
            """
            ## About This Project
            
            This is a real-time vehicle detection system built with:
            - **Model:** YOLOv8n (Nano)
            - **Framework:** Ultralytics
            - **Deployment:** Gradio + Docker + Hugging Face Spaces
            - **MLOps:** GitHub Actions CI/CD, WandB for experiment tracking
            
            ### Detected Classes
            1. Auto
            2. Bus
            3. Car
            4. LCV (Light Commercial Vehicle)
            5. Motorcycle
            6. Multiaxle
            7. Tractor
            8. Truck
            
            ### Performance
            - **Speed:** ~50 FPS on GPU, ~10 FPS on CPU
            - **Accuracy:** mAP@0.5 > 85%
            
            ### Source Code
            Check out the [GitHub Repository](https://github.com/yourusername/vehicle-detection-mlops)
            """
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)