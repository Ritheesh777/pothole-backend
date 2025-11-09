import cv2
import numpy as np
from ultralytics import YOLO
import logging
import os
from PIL import Image
import torch
# --- Import Depth Estimator (Assumes it's in backend/utils/depth_estimation.py) ---
from utils.depth_estimation import depth_estimator 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PotholeDetector:
    def __init__(self, model_path=None):
        """
        Initialize YOLO model for pothole detection
        """
        self.model = None
        
        # --- CRITICAL CHANGE: POINT TO YOUR TRAINED MODEL WEIGHTS ---
        # This path assumes 'best.pt' is in the backend/models/ directory.
        DEFAULT_MODEL_PATH = 'D:/pothole-detection-project/backend/models/best.pt' 
        self.model_path = model_path or DEFAULT_MODEL_PATH
        
        # --- PRODUCTION THRESHOLDS ---
        # Increased confidence threshold to reduce false positives (lowering False Alarms)
        self.confidence_threshold = 0.25  
        self.iou_threshold = 0.45
        self.load_model()
    
    def load_model(self):
        """Load YOLOv8 model"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Custom model not found at {self.model_path}. Falling back to pre-trained.")
                self.model = YOLO('yolov8m.pt') # Fallback
            else:
                logger.info(f"Loading custom model from {self.model_path}")
                self.model = YOLO(self.model_path)
            
            # Set model to evaluation mode and GPU if available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(device).model.eval()
            
            logger.info(f"YOLOv8 model loaded successfully on {device}.")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def preprocess_image(self, image):
        """Preprocess image for YOLO detection (Ensures correct color space and NumPy array)"""
        if isinstance(image, np.ndarray):
            # Convert BGR (OpenCV default) to RGB (YOLO preferred for internal processing)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        return image
    
    def detect_potholes(self, image, return_image=False):
        """Detect potholes in the given image and return raw bounding box data."""
        try:
            processed_image = self.preprocess_image(image)
            
            results = self.model(
                processed_image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            annotated_image = None
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy() # xyxy format
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                        
                        detections.append({
                            'bbox_xyxy': [int(box[0]), int(box[1]), int(box[2]), int(box[3])], # xyxy format
                            'confidence': float(conf),
                            'class_id': int(cls),
                        })
                
                if return_image:
                    annotated_image = result.plot() # BGR output array
            
            return {
                'success': True,
                'detections': detections,
                'annotated_image': annotated_image,
                'image_shape': processed_image.shape,
                'raw_image_array': processed_image
            }
            
        except Exception as e:
            logger.error(f"Error during YOLO detection: {e}")
            return {'success': False, 'error': str(e), 'detections': []}

    
    def process_image_for_api(self, image_array, latitude=None, longitude=None):
        """
        Complete processing pipeline: YOLO Detect -> Estimate Depth -> Structure Data
        """
        try:
            # 1. DETECT Potholes using YOLO
            detection_results = self.detect_potholes(image_array, return_image=True)
            
            if not detection_results['success']:
                return detection_results
            
            raw_detections = detection_results['detections']
            processed_potholes = []
            
            # 2. PROCESS each detection for Severity/Depth
            for detection in raw_detections:
                x1, y1, x2, y2 = detection['bbox_xyxy']
                
                # --- INTEGRATING DEPTH ESTIMATION FOR SEVERITY ---
                severity_info = depth_estimator.analyze_pothole_depth(
                    image_array, 
                    (x1, y1, x2, y2)
                )
                
                # 3. STRUCTURE FINAL OUTPUT DATA (Frontend compatibility)
                pothole_data = {
                    'bbox_xyxy': detection['bbox_xyxy'],
                    'confidence': round(detection['confidence'], 4),
                    'severity': severity_info['severity'],
                    'depth_estimate_m': round(severity_info['depth'], 4) if severity_info['depth'] is not None else None,
                    'priority': severity_info['priority']
                }
                
                # Add location if provided
                if latitude is not None and longitude is not None:
                    pothole_data['latitude'] = latitude
                    pothole_data['longitude'] = longitude
                
                processed_potholes.append(pothole_data)
            
            logger.info(f"Successfully processed {len(processed_potholes)} potholes with severity analysis.")
            
            return {
                'success': True,
                'potholes': processed_potholes,
                'total_filtered': len(processed_potholes),
                'annotated_image': detection_results['annotated_image']
            }
            
        except Exception as e:
            logger.error(f"Critical error in processing pipeline: {e}")
            return {
                'success': False,
                'error': f"Critical processing error: {e}",
                'potholes': []
            }

# Global detector instance (only initialized once)
pothole_detector = PotholeDetector()
