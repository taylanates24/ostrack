import cv2
import os

def resize_video_opencv(input_path, output_path, new_size):
    """
    Resizes a video to a new size using OpenCV.

    :param input_path: Path to the input video file.
    :param output_path: Path to save the resized video file.
    :param new_size: A tuple (width, height) for the new video size.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input video not found at {input_path}")
        return

    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get original video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Original video size: {original_width}x{original_height}, FPS: {fps}")

    # Define the codec and create VideoWriter object
    # MP4V is a good codec for .mp4 files.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, new_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"frame_drone.jpg", frame)
        # Resize the frame
        resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        # Write the resized frame
        out.write(resized_frame)

    # Release everything when job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video successfully resized and saved to {output_path}")


if __name__ == '__main__':
    # --- Configuration ---
    input_video_path = "drone/drone_red_object_1.mp4"  # Replace with your input video path
    output_video_path = "drone_red_object_1_resized.mp4"  # Replace with your desired output path
    width = 1280  # Replace with your desired width
    height = 720  # Replace with your desired height
    # ---------------------

    new_dimensions = (width, height)
    resize_video_opencv(input_video_path, output_video_path, new_dimensions)


#ffmpeg -i video.mp4 video_reencoded.mp4