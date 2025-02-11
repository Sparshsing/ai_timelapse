# download model
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp

import urllib.request
import os

def download_face_landmark_model():
    # Check if file already exists
    if not os.path.exists('shape_predictor_68_face_landmarks.dat'):
        print("Downloading face landmarks model...")
        # The raw file URL from GitHub
        model_url = 'https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2'
        # Download the compressed file
        urllib.request.urlretrieve(model_url, 'shape_predictor_68_face_landmarks.dat.bz2')
        
        # Decompress the file
        import bz2
        with bz2.BZ2File('shape_predictor_68_face_landmarks.dat.bz2') as fr, \
            open('shape_predictor_68_face_landmarks.dat', 'wb') as fw:
            fw.write(fr.read())
        
        # Remove the compressed file
        os.remove('shape_predictor_68_face_landmarks.dat.bz2')
        print("Download complete!")
    else:
        print("Model file already exists")


def get_landmarks(image, detector, predictor):
    """
    Detect facial landmarks in an image using dlib.
    Returns a numpy array of shape (68, 2) with (x, y) coordinates.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        raise Exception("No faces detected in the image.")
    # For simplicity, use the first detected face
    face = faces[0]
    shape = predictor(gray, face)
    coords = np.array([[pt.x, pt.y] for pt in shape.parts()])
    return coords


def compute_similarity_transform(source, target):
    """
    Compute a similarity (affine) transformation matrix that maps
    the source landmarks to the target landmarks using a Procrustes analysis.
    
    Returns a 2x3 affine transformation matrix M such that:
       target ≈ M * source
    """
    # Compute centroids
    src_mean = np.mean(source, axis=0)
    tgt_mean = np.mean(target, axis=0)
    
    # Center the points
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    
    # Compute the scaling factors (Frobenius norm)
    src_norm = np.linalg.norm(src_centered)
    tgt_norm = np.linalg.norm(tgt_centered)
    
    # Normalize the centered coordinates
    src_normalized = src_centered / src_norm
    tgt_normalized = tgt_centered / tgt_norm
    
    # Compute rotation using SVD on the covariance matrix
    U, _, Vt = np.linalg.svd(np.dot(src_normalized.T, tgt_normalized))
    R = np.dot(Vt.T, U.T)
    
    # Compute the scale factor
    scale = tgt_norm / src_norm
    
    # Compute the translation
    T = tgt_mean - scale * np.dot(src_mean, R)
    
    # Build the 2x3 affine transform matrix
    M = np.zeros((2, 3))
    M[:2, :2] = scale * R
    M[:, 2] = T
    
    return M


def rotate_image_with_padding(image, M, background_color=(255, 255, 255)):
    """
    Rotate an image without cropping and with white background padding.
    
    Args:
        image: Input image as numpy array
        M: 2x3 transformation matrix
        background_color: Color to fill the background (default: white)
        
    Returns:
        Rotated image with padding
    """
    # Get the image size
    height, width = image.shape[:2]
    
    # Calculate the new image dimensions
    # Get the corner points of the image
    corners = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # Transform the corners
    transformed_corners = cv2.transform(corners.reshape(1, -1, 2), M).reshape(-1, 2)
    
    # Get the new image dimensions
    min_x = np.floor(transformed_corners[:, 0].min()).astype(int)
    max_x = np.ceil(transformed_corners[:, 0].max()).astype(int)
    min_y = np.floor(transformed_corners[:, 1].min()).astype(int)
    max_y = np.ceil(transformed_corners[:, 1].max()).astype(int)
    
    # Calculate new width and height
    new_width = max_x - min_x
    new_height = max_y - min_y
    
    # Adjust the transformation matrix to account for the padding
    M_adjusted = M.copy()
    M_adjusted[0, 2] -= min_x
    M_adjusted[1, 2] -= min_y
    
    # Create a white background image
    if len(image.shape) == 3:  # Color image
        background = np.full((new_height, new_width, 3), background_color, dtype=np.uint8)
    else:  # Grayscale image
        background = np.full((new_height, new_width), background_color[0], dtype=np.uint8)
    
    # Apply the transformation
    result = cv2.warpAffine(
        image,
        M_adjusted,
        (new_width, new_height),
        borderMode=cv2.BORDER_TRANSPARENT,
        borderValue=background_color
    )
    
    # Blend the rotated image with the background
    mask = cv2.warpAffine(
        np.ones_like(image) * 255,
        M_adjusted,
        (new_width, new_height),
        borderMode=cv2.BORDER_TRANSPARENT,
        borderValue=0
    )
    mask = mask.astype(bool)
    background[mask] = result[mask]
    
    return background

# Example usage:
# aligned_img = rotate_image_with_padding(target_img, M)

def align_faces(target_img: np.ndarray, ref_img: np.ndarray) -> np.ndarray:

    # --- 2. Initialize Dlib Face Detector and Landmark Predictor ---
    detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_68_face_landmarks.dat"  # update path if needed
    predictor = dlib.shape_predictor(predictor_path)

    # --- 3. Detect Landmarks in Both Images ---
    ref_landmarks = get_landmarks(ref_img, detector, predictor)
    target_landmarks = get_landmarks(target_img, detector, predictor)

    # --- 4. Compute the Similarity Transformation ---
    # This gives a global alignment (rotation, scaling, translation)
    M = compute_similarity_transform(target_landmarks, ref_landmarks)

    # Warp the target image using the computed affine transformation.
    # The output size is chosen to match the reference image.
    h_ref, w_ref = ref_img.shape[:2]
    # aligned_img = cv2.warpAffine(target_img, M, (w_ref, h_ref))
    aligned_img = rotate_image_with_padding(target_img, M)
    return aligned_img


    # # --- 5. Non-Rigid (Piecewise) Warping for Texture Normalization ---
    # # Using the detected landmarks as control points, estimate a piecewise affine
    # # transform from the target landmarks to the reference landmarks.
    # tform = PiecewiseAffineTransform()
    # tform.estimate(target_landmarks, ref_landmarks)

    # # Warp the target image using the piecewise affine transform.
    # # Note: warp() returns a float image with values in [0,1], so we convert back.
    # warped_img = warp(target_img, tform, output_shape=(h_ref, w_ref))
    # warped_img = (warped_img * 255).astype(np.uint8)


download_face_landmark_model()
