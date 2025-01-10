import cv2
import mediapipe as mp
import numpy as np
from rembg import remove
from pathlib import Path
import logging
from typing import Tuple, Optional
from natsort import natsorted

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from the specified path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array
        
    Raises:
        ImageProcessingError: If image cannot be loaded
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        image = cv2.imread(str(image_path))
        if image is None:
            raise ImageProcessingError("Failed to load image")
            
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise ImageProcessingError(f"Error loading image: {str(e)}")

def detect_face(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect face in the image using MediaPipe Face Detection.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Tuple of (x, y, width, height) of the face bounding box or None if no face detected
        
    Raises:
        ImageProcessingError: If face detection fails
    """
    try:
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(
            model_selection=1,  # Use full range model
            min_detection_confidence=0.5
        ) as face_detection:
            results = face_detection.process(image)
            
            if not results.detections:
                logger.warning("No face detected in the image")
                return None
                
            # Get the first detected face
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            ih, iw, _ = image.shape
            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)
            
            return (x, y, w, h)
    except Exception as e:
        raise ImageProcessingError(f"Error in face detection: {str(e)}")


def draw_face_bbox(
    image: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding box around detected face.
    
    Args:
        image: Input image
        face_bbox: Face bounding box (x, y, width, height)
        color: BGR color tuple for the box
        thickness: Line thickness
        
    Returns:
        Image with drawn bounding box
    """
    try:
        x, y, w, h = face_bbox
        image_copy = image.copy()
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)
        return image_copy
    except Exception as e:
        raise ImageProcessingError(f"Error drawing bounding box: {str(e)}")


def crop_and_center_face(
    image: np.ndarray,
    face_bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Crop and center the face in the image with padding.
    Face will take up half the width and height of the final image.
    
    Args:
        image: Input image
        face_bbox: Face bounding box (x, y, width, height)
        
    Returns:
        Cropped and centered image with padding if necessary
    """
    try:
        x, y, w, h = face_bbox
        ih, iw, _ = image.shape
        
        # Calculate target size (face should be half the final image)
        target_size = max(w * 2, h * 2)
        
        # Calculate center points
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Calculate crop boundaries
        left = center_x - target_size // 2
        top = center_y - target_size // 2
        right = left + target_size
        bottom = top + target_size
        
        # Create white canvas
        result = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
        
        # Calculate source and destination regions for copying
        src_left = max(0, left)
        src_top = max(0, top)
        src_right = min(iw, right)
        src_bottom = min(ih, bottom)
        
        dst_left = max(0, -left)
        dst_top = max(0, -top)
        
        # Copy valid region
        result[
            dst_top:dst_top + (src_bottom - src_top),
            dst_left:dst_left + (src_right - src_left)
        ] = image[src_top:src_bottom, src_left:src_right]
        
        return result
    except Exception as e:
        raise ImageProcessingError(f"Error in cropping and centering: {str(e)}")

def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to specified size.
    
    Args:
        image: Input image
        size: Target size as (width, height)
        
    Returns:
        Resized image
    """
    try:
        return cv2.resize(
            image,
            size,
            interpolation=cv2.INTER_LANCZOS4
        )
    except Exception as e:
        raise ImageProcessingError(f"Error in image resizing: {str(e)}")

def remove_background(
    image: np.ndarray,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Remove background from image using Rembg and replace with specified color.
    
    Args:
        image: Input image
        background_color: RGB color tuple for the background
        
    Returns:
        Image with background replaced by specified color
    """
    try:
        # Remove background
        output = remove(image)
        
        # Create background color image
        bg = np.full(output.shape, background_color + (255,), dtype=np.uint8)
        
        # Blend the image with background using alpha channel
        alpha = output[:, :, 3:] / 255.0
        foreground = output[:, :, :3] * alpha
        background = bg[:, :, :3] * (1 - alpha)
        
        # Combine foreground and background
        result = (foreground + background).astype(np.uint8)
        
        return result
    except Exception as e:
        raise ImageProcessingError(f"Error in background removal: {str(e)}")


def process_face_image(
    image_path: str,
    output_path: str,
    target_size: Tuple[int, int] = (1000, 1000),
    background_color: Tuple[int, int, int] = (255, 255, 255),
    save_bbox_preview: bool = False
) -> None:
    """
    Main function to process the face image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save the processed image
        target_size: Final image size (width, height)
        background_color: RGB color tuple for background after removal
        save_bbox_preview: If True, saves an additional image with drawn bounding box
    """
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = load_image(image_path)
        logger.info("Image loaded successfully")
        
        # Detect face
        face_bbox = detect_face(image)
        if face_bbox is None:
            raise ImageProcessingError("No face detected in the image")
        logger.info("Face detected successfully")
        
        # Save bbox preview if requested
        if save_bbox_preview:
            preview_img = draw_face_bbox(image, face_bbox)
            preview_path = str(Path(output_path).with_stem(f"{Path(output_path).stem}_bbox"))
            cv2.imwrite(
                preview_path,
                cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR)
            )
            logger.info(f"Bounding box preview saved to: {preview_path}")
        
        # Crop and center face
        centered_image = crop_and_center_face(image, face_bbox)
        logger.info("Image cropped and centered")
        
        # Resize image
        resized_image = resize_image(centered_image, target_size)
        logger.info("Image resized")
        
        # Remove background
        final_image = remove_background(resized_image, background_color)
        logger.info("Background removed")
        
        # Save result as JPG
        cv2.imwrite(
            output_path,
            cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        logger.info(f"Processed image saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise


def process_directory(
    input_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (1000, 1000),
    background_color: Tuple[int, int, int] = (255, 255, 255),
    save_bbox_preview: bool = False
) -> None:
    """
    Process all images in a directory.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for processed images
        target_size: Final image size (width, height)
        background_color: RGB color tuple for background after removal
        save_bbox_preview: If True, saves additional images with drawn bounding boxes
    """
    try:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        # Sort files naturally
        image_files = natsorted(image_files, key=lambda x: x.name)
        
        total_files = len(image_files)
        logger.info(f"Found {total_files} images to process")
        
        # Process each image
        for idx, image_path in enumerate(image_files, 1):
            try:
                output_path = output_dir / f"{image_path.stem}_processed.jpg"
                logger.info(f"Processing {idx}/{total_files}: {image_path.name}")
                
                process_face_image(
                    str(image_path),
                    str(output_path),
                    target_size,
                    background_color,
                    save_bbox_preview
                )
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {str(e)}")
                continue
        
        logger.info("Directory processing completed")
        
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    try:
        process_face_image(
            "input.jpg",
            "output.png",
            background_color=(255, 255, 255),  # White background
            save_bbox_preview=True
            )
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")

