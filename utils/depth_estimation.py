import cv2
import numpy as np
import torch
import torch.nn.functional as F
import logging
import random 

# NOTE: You MUST install the 'timm' library for MiDaS to load correctly: pip install timm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MiDaS Integration Setup ---
midas = None
transform = None
midas_device = 'cpu'

try:
    logger.info("Attempting to load MiDaS model from PyTorch Hub...")
    # Attempt to load actual MiDaS small model
    # Requires 'timm' library and internet access for the first run
    midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=True)
    midas_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(midas_device)
    midas.eval()
    
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    # NOTE: The transforms object is complex, but MiDaS uses specific transforms
    if hasattr(midas_transforms, 'small_transform'):
        transform = midas_transforms.small_transform
    else:
        transform = None
        
    logger.info(f"MiDaS model loaded successfully on {midas_device}.")
    
except Exception as e:
    logger.warning(f"MiDaS model loading failed. Using placeholder depth. Error: {e}")

class DepthEstimator:
    def __init__(self):
        self.midas = midas
        self.transform = transform
        self.device = midas_device
        self.is_placeholder = self.midas is None
        
        if self.is_placeholder:
            logger.warning("Depth Estimator is running in PLACEHOLDER/MOCK mode.")
            # If transform is None, define a basic mock transform
            if self.transform is None:
                 self.transform = self._mock_transform

    def _mock_transform(self, image_array):
        """Mock transform to prevent crash if MiDaS loading fails entirely."""
        if len(image_array.shape) == 3:
            return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        return image_array

    def _placeholder_depth_estimation(self, image_array, bbox):
        """
        Fallback: Placeholder depth estimation using simple image analysis (brightness/texture).
        """
        x1, y1, x2, y2 = bbox
        # Crop region of interest (safe access)
        roi = image_array[int(y1):int(y2), int(x1):int(x2)]
        
        if roi.size == 0:
            return {'depth_value': 0.0, 'depth_score': 0.0, 'method': 'placeholder'}

        # Convert to grayscale for statistical analysis
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Logic: Darker regions with high variation suggest a hole/shadow.
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Score normalized from 0 (lightest/smoothest) to 1 (darkest/highest variation)
        depth_score = (255 - mean_intensity) / 255.0 * (std_intensity / 50.0)
        depth_score = np.clip(depth_score, 0.05, 0.8) 
        
        estimated_depth_m = depth_score * 0.3 
        estimated_depth_m = max(0.01, estimated_depth_m + random.uniform(-0.01, 0.01))
        
        return {
            'depth_value': float(estimated_depth_m),
            'depth_score': float(depth_score),
            'method': 'placeholder'
        }

    def _midas_depth_estimation(self, image_array, bbox):
        """
        Actual MiDaS Depth Estimation Logic.
        """
        x1, y1, x2, y2 = bbox
        pothole_region = image_array[int(y1):int(y2), int(x1):int(x2)]

        if pothole_region.size == 0 or self.transform is None:
            return {'depth_value': 0.0, 'depth_score': 0.0, 'method': 'midas_failed'}

        # MiDaS works better with RGB image arrays
        pothole_region_rgb = cv2.cvtColor(pothole_region, cv2.COLOR_BGR2RGB)
            
        input_batch = self.transform(pothole_region_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=pothole_region.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        # Invert and normalize depth map (lower value = farther/deeper)
        max_depth = np.max(depth_map)
        min_depth = np.min(depth_map)
        
        if max_depth == min_depth:
            normalized_depth = np.zeros_like(depth_map)
        else:
            normalized_depth = (max_depth - depth_map) / (max_depth - min_depth)
        
        avg_depth_score = np.mean(normalized_depth)

        # Arbitrary scale: Max depth of 30cm (0.3m)
        estimated_depth_m = avg_depth_score * 0.3 
            
        return {
            'depth_value': float(estimated_depth_m),
            'depth_score': float(avg_depth_score),
            'method': 'midas'
        }


    def analyze_pothole_depth(self, image_array, pothole_bbox):
        """
        Analyze depth specifically for a detected pothole (Public API for PotholeDetector)
        """
        try:
            if self.is_placeholder:
                depth_result = self._placeholder_depth_estimation(image_array, pothole_bbox)
            else:
                depth_result = self._midas_depth_estimation(image_array, pothole_bbox)

            depth_value = depth_result.get('depth_value')
            
            if depth_value is not None:
                # --- Severity Classification (Client Logic) ---
                # Based on estimated depth (0.3m max)
                if depth_value > 0.15: # Deeper than 15 cm
                    severity = 'CRITICAL'
                    priority = 4
                elif depth_value > 0.08: # Deeper than 8 cm
                    severity = 'SEVERE'
                    priority = 3
                elif depth_value > 0.04: # Deeper than 4 cm
                    severity = 'MODERATE'
                    priority = 2
                else:
                    severity = 'MINOR'
                    priority = 1
                
                return {
                    'depth': round(depth_value, 4),
                    'severity': severity,
                    'priority': priority, 
                    'confidence': round(depth_result.get('depth_score', 0.5), 4) 
                }
        except Exception as e:
            logger.error(f"Failed during depth analysis: {e}")
            
        return {
            'depth': None,
            'severity': 'UNKNOWN',
            'priority': 0,
            'confidence': 0.0
        }

# Global depth estimator instance (initialized once)
depth_estimator = DepthEstimator()
