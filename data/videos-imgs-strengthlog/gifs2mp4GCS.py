from google.cloud import storage
import imageio
from moviepy.editor import ImageSequenceClip
import os

# Initialize GCS client
storage_client = storage.Client()

# Define your GCS buckets and folders
source_bucket_name = 'genai-elena'
source_folder = 'multimodal-video/gym-exercises/exercises-videos/strengthlog/'
target_bucket_name = 'genai-elena'
target_folder = 'multimodal-video/gym-exercises/exercises-videos/'

# Reference to the source and target buckets
source_bucket = storage_client.bucket(source_bucket_name)
target_bucket = storage_client.bucket(target_bucket_name)

# List all GIF files in the source folder
blobs = source_bucket.list_blobs(prefix=source_folder)
gif_files = [blob for blob in blobs if blob.name.endswith('.gif')]

for blob in gif_files:
    # Download GIF file
    gif_path = blob.name.split('/')[-1]  # Extract file name
    blob.download_to_filename(gif_path)

    # Convert GIF to MP4
    output_path = gif_path.replace('.gif', '.mp4')
    reader = imageio.get_reader(gif_path)
    fps = 24  # Assuming a default FPS
    frames = [frame for frame in reader]
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path)
    clip.close()

    # Upload MP4 to target GCS folder
    blob_target = target_bucket.blob(f"{target_folder}{output_path}")
    blob_target.upload_from_filename(output_path)

    # Clean up: delete local files
    os.remove(gif_path)
    os.remove(output_path)

print("Conversion and upload completed.")