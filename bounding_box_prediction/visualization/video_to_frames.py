from datetime import timedelta
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import bbox_from_csv as bfc
import imageio#.v2 as imageio


### Save the frames of a video as image files ###

def save_frames(video_path, output_dir):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if the video file was opened successfully
    if not video.isOpened():
        print("Error opening video file")
        return
    
    # Get some properties of the video
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and save each frame of the video
    for frame_index in range(frame_count):
        # Set the current frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        # Read the frame
        ret, frame = video.read()
        
        if not ret:
            print(f"Error reading frame {frame_index}")
            continue
        
        # Construct the output file path
        output_path = os.path.join(output_dir, f"{frame_index:04d}.png")

        # Check if the file already exists
        if os.path.exists(output_path):
            os.remove(output_path)  # Delete the existing file
        
        # Save the frame as an image file
        cv2.imwrite(output_path, frame)
        
        # Display progress
        print(f"Saved frame {frame_index}/{frame_count}")
    
    # Release the video file
    video.release()
    print("All frames saved successfully!")

# Example usage
""" video_path = "visualization/video_1.mp4"
output_dir = "visualization/video_1_frames_png"
save_frames(video_path, output_dir) """





### Draw the bounding boxes on the frames ###

def draw_bbox(frames_dir, output_dir, bbox_csv_dir, number_of_lines):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # import the bbox array
    pred_array, true_array, name_array = bfc.bbox_array_from_csv(bbox_csv_dir, number_of_lines)

    big_count = 0
    small_count = 0
    frame_count = 0

    # Iterate over the frames and find the bounding box locations
    for i in name_array:
        for j in range(16):
            frame_number = int(i[j].replace('.png', ''))

            # Get the bounding box coordinates
            if small_count == 16:
                big_count += 1
                small_count = 0  

            if frame_number >= 8000:
                continue

            # Find the frame in the frames directory
            frame_name = i[j] #f"{frame_number:04d}.png"

            # Check if the frame already has a bounding box in the output directory
            output_path = os.path.join(output_dir, frame_name)
            if os.path.exists(output_path):
                frame = cv2.imread(output_path)

                # Convert the bbox coordinates to floats
                float_true_bbox = true_array[big_count, small_count, :].astype(float)
                float_pred_bbox = pred_array[big_count, small_count, :].astype(float)
                small_count += 1

                x0_true, y0_true, w_true, h_true = float_true_bbox
                x0_pred, y0_pred, w_pred, h_pred = float_pred_bbox

                # Draw the true bounding box on the frame
                pt1_true = (int(x0_true - w_true/2), int(y0_true - h_true/2))
                pt2_true = (int(x0_true + w_true/2), int(y0_true + h_true/2))
                cv2.rectangle(frame, pt1_true, pt2_true, (0, 255, 0), 2)

                # Draw the predicted bounding box on the frame
                pt1_pred = (int(x0_pred - w_pred/2), int(y0_pred - h_pred/2))
                pt2_pred = (int(x0_pred + w_pred/2), int(y0_pred + h_pred/2))
                cv2.rectangle(frame, pt1_pred, pt2_pred, (0, 0, 255), 2)               

                os.remove(output_path)  # Delete the existing file

                # Construct the output file path
                output_path = os.path.join(output_dir, frame_name)
                
                # Save the frame as an image file
                cv2.imwrite(output_path, frame)
                
                # Display progress
                print(f"Saved frame on annotated frame {frame_name}")

                frame_count += 1

                continue

            frame_path = os.path.join(frames_dir, frame_name)
            frame = cv2.imread(frame_path)

            # Convert the bbox coordinates to floats
            float_true_bbox = true_array[big_count, small_count, :].astype(float)
            float_pred_bbox = pred_array[big_count, small_count, :].astype(float)
            small_count += 1

            x0_true, y0_true, w_true, h_true = float_true_bbox
            x0_pred, y0_pred, w_pred, h_pred = float_pred_bbox

            # Draw the true bounding box on the frame
            pt1_true = (int(x0_true - w_true/2), int(y0_true - h_true/2))
            pt2_true = (int(x0_true + w_true/2), int(y0_true + h_true/2))
            cv2.rectangle(frame, pt1_true, pt2_true, (0, 255, 0), 2)

            # Draw the predicted bounding box on the frame
            pt1_pred = (int(x0_pred - w_pred/2), int(y0_pred - h_pred/2))
            pt2_pred = (int(x0_pred + w_pred/2), int(y0_pred + h_pred/2))
            cv2.rectangle(frame, pt1_pred, pt2_pred, (0, 0, 2555), 2) 

            # Construct the output file path
            output_path = os.path.join(output_dir, frame_name)

            # Check if the file already exists
            if os.path.exists(output_path):
                os.remove(output_path)  # Delete the existing file
            
            # Save the frame as an image file
            cv2.imwrite(output_path, frame)
            
            # Display progress
            print(f"Saved frame {frame_name}")

            frame_count += 1
            if frame_count >= number_of_lines*16:
                break

# Example usage
""" frames_dir = "visualization/video_1_frames_png"
output_dir = "visualization/video_1_bbox"
bbox_csv_dir = "visualization/bbox_results_1e-05_512(1).csv"
draw_bbox(frames_dir, output_dir, bbox_csv_dir, 47) """





### Convert the frames to a GIF ###

def frames_to_gif(frames_dir, output_path, fps):
    # Get the path of all the image files
    image_paths = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir)]
    
    # Sort the image paths alphabetically
    image_paths.sort()
    
    # Read the images
    images = [imageio.imread(image_path) for image_path in image_paths]
    
    # Save the frames as a GIF
    imageio.mimsave(output_path, images, duration=ms, quality=10, macro_block_size=None)
    
    print("GIF created successfully!")

# Example usage
""" frames_dir = "visualization/video_1_bbox_short"
output_path = "visualization/video_1_pred_bbox_short.gif"
ms = 20
frames_to_gif(frames_dir, output_path, ms) """
