import cv2
import numpy as np

def preprocess_region(image_pil):
    """
    Applies grayscale, denoising, binarization, and dilation to a cropped image region.
    """
    sample = np.array(image_pil)
    
    # Handle RGB conversion if necessary
    if len(sample.shape) == 3:
        gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    else:
        gray = sample

    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Resize logic
    target_height = 120
    if binary.shape[0] > 0:
        scale = target_height / binary.shape[0]
        resized = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        resized = binary

    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.dilate(resized, kernel, iterations=1)
    
    return processed