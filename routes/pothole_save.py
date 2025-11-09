from flask import Blueprint, request, jsonify
import os
import base64
from werkzeug.utils import secure_filename
from database.db import db_manager
import json
import logging

pothole_save_bp = Blueprint('pothole_save', __name__)
logger = logging.getLogger(__name__)

# Create images directory if it doesn't exist
IMAGES_DIR = 'pothole_images'
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

@pothole_save_bp.route('/api/save_pothole', methods=['POST'])
def save_pothole():
    """Save pothole data and image to database and file system"""
    try:
        # Get pothole data from form
        pothole_data_json = request.form.get('pothole_data')
        if not pothole_data_json:
            return jsonify({
                'success': False,
                'error': 'No pothole data provided'
            }), 400
        
        pothole_data = json.loads(pothole_data_json)
        
        # Handle image upload
        image_url = None
        if 'pothole_image' in request.files:
            image_file = request.files['pothole_image']
            if image_file.filename:
                # Create secure filename
                filename = secure_filename(f"{pothole_data['pothole_id']}.jpg")
                filepath = os.path.join(IMAGES_DIR, filename)
                
                # Save image file
                image_file.save(filepath)
                image_url = f"/images/{filename}"
                logger.info(f"Image saved: {filepath}")
        
        # Prepare data for database
        db_pothole_data = {
            'pothole_id': pothole_data['pothole_id'],
            'survey_name': pothole_data['survey_name'],
            'detection_type': pothole_data.get('detection_type', 'capture'),
            'latitude': pothole_data['latitude'],
            'longitude': pothole_data['longitude'],
            'confidence': pothole_data['confidence'],
            'depth': pothole_data.get('depth'),
            'bbox_x': pothole_data.get('bbox', [0, 0, 0, 0])[0],
            'bbox_y': pothole_data.get('bbox', [0, 0, 0, 0])[1],
            'bbox_width': pothole_data.get('bbox', [0, 0, 0, 0])[2],
            'bbox_height': pothole_data.get('bbox', [0, 0, 0, 0])[3],
            'area_description': pothole_data.get('area', 'Unknown'),
            'image_url': image_url
        }
        
        # Save to database
        result = db_manager.insert_pothole(db_pothole_data)
        
        return jsonify({
            'success': True,
            'message': 'Pothole saved successfully',
            'pothole_id': pothole_data['pothole_id'],
            'database_id': result['id'],
            'image_url': image_url
        }), 200
        
    except Exception as e:
        logger.error(f"Error saving pothole: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to save pothole: {str(e)}'
        }), 500

@pothole_save_bp.route('/images/<filename>')
def serve_image(filename):
    """Serve uploaded pothole images"""
    try:
        from flask import send_from_directory
        return send_from_directory(IMAGES_DIR, filename)
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return jsonify({'error': 'Image not found'}), 404