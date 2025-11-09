from flask import Blueprint, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io
import base64
import logging
import uuid
from datetime import datetime

# --- IMPORTANT IMPORTS ---
# Assumes models/yolov8_model.py is updated
from models.yolov8_model import pothole_detector
# Assumes utils/depth_estimation.py is updated
from utils.depth_estimation import depth_estimator 
from database.db import db_manager 

detect_bp = Blueprint('detect', __name__)
logger = logging.getLogger(__name__)

@detect_bp.route('/detect', methods=['POST'])
def detect_potholes():
    """
    Detect potholes in uploaded image (Single detection)
    
    Expected form data:
    - image: Image file
    - latitude: Optional latitude coordinate
    - longitude: Optional longitude coordinate
    - survey_name: Optional survey name
    """
    try:
        # 1. Input Validation and Data Retrieval
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No image file selected'}), 400
        
        # Get optional location data and survey name
        latitude = request.form.get('latitude', type=float)
        longitude = request.form.get('longitude', type=float)
        survey_name = request.form.get('survey_name', default='Quick Scan') 
        
        # Read and prepare image array
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # 2. RUN COMPLETE DETECTION & SEVERITY PIPELINE
        # pothole_detector.process_image_for_api handles detection and depth/severity calculation
        detection_results = pothole_detector.process_image_for_api(
            image_array, latitude, longitude
        )
        
        if not detection_results['success']:
            return jsonify(detection_results), 500
        
        processed_potholes = detection_results['potholes']
        saved_potholes_count = 0
        
        # 3. SAVE TO DATABASE (Only if location is provided)
        if latitude is not None and longitude is not None:
            for pothole in processed_potholes:
                try:
                    # Bounding Box: Convert XYXY (YOLO format) to XYWH (DB format)
                    x1, y1, x2, y2 = pothole['bbox_xyxy']
                    
                    db_pothole_data = {
                        'pothole_id': str(uuid.uuid4()), # Generate Unique ID
                        'survey_name': survey_name,
                        'latitude': pothole['latitude'],
                        'longitude': pothole['longitude'],
                        'confidence': pothole['confidence'],
                        # Fields provided by depth_estimation.py
                        'depth': pothole.get('depth_estimate_m'),
                        'severity': pothole.get('severity'),
                        'priority': pothole.get('priority'),
                        
                        'bbox_x': x1,
                        'bbox_y': y1,
                        'bbox_width': x2 - x1,
                        'bbox_height': y2 - y1,
                        'area_description': f"{pothole['severity']} Pothole",
                        'detection_type': 'live' if 'live' in survey_name.lower() else 'capture'
                    }
                    
                    saved_pothole = db_manager.insert_pothole(db_pothole_data)
                    pothole['database_id'] = saved_pothole['id']
                    pothole['detected_at'] = saved_pothole['detected_at'].isoformat()
                    saved_potholes_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to save pothole to database: {e}")
                    pothole['database_error'] = str(e)
        
        # 4. PREPARE FINAL RESPONSE (Frontend compatibility ensured)
        response_data = {
            'success': True,
            'message': f'Detected {len(processed_potholes)} potholes with severity analysis',
            'potholes': processed_potholes,
            'total_detections': detection_results['total_filtered'],
            'filtered_detections': detection_results['total_filtered'], 
            'location_provided': latitude is not None and longitude is not None,
            'saved_to_database': saved_potholes_count
        }
        
        # Add annotated image in base64 format for frontend display
        if detection_results.get('annotated_image') is not None:
            annotated_image = detection_results['annotated_image']
            if isinstance(annotated_image, np.ndarray):
                # Convert BGR (YOLO plot output) -> RGB -> JPEG for base64
                pil_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                response_data['annotated_image'] = f"data:image/jpeg;base64,{img_str}"
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Critical error in detection endpoint: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@detect_bp.route('/detect/batch', methods=['POST'])
def detect_potholes_batch():
    # Kept for compatibility. Returns an error message.
    return jsonify({
        'success': False,
        'error': 'Batch detection is complex and currently not fully implemented. Use /detect endpoint for single images.'
    }), 501

@detect_bp.route('/detect/health', methods=['GET'])
def detection_health():
    """
    Health check for detection service
    """
    try:
        # Test model availability (model is None if it failed to load)
        model_status = pothole_detector.model is not None
        # Test depth model status (is_placeholder=False means MiDaS successfully loaded)
        depth_status = depth_estimator.is_placeholder is False 
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'components': {
                'yolov8_model': model_status,
                'depth_estimator': depth_status,
                'database': True 
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e)
        }), 500
