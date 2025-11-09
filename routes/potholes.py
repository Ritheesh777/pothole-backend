from flask import Blueprint, request, jsonify
import logging
from database.db import db_manager
from datetime import datetime, timedelta

potholes_bp = Blueprint('potholes', __name__)
logger = logging.getLogger(__name__)

@potholes_bp.route('/potholes', methods=['GET'])
def get_all_potholes():
    """
    Get all potholes with optional filtering
    
    Query parameters:
    - limit: Maximum number of potholes to return (default: 1000)
    - since: ISO date string to filter potholes since that date
    - min_confidence: Minimum confidence threshold (0.0-1.0)
    """
    try:
        # Get query parameters
        limit = request.args.get('limit', default=1000, type=int)
        since = request.args.get('since')
        min_confidence = request.args.get('min_confidence', default=0.0, type=float)
        
        # Validate parameters
        limit = min(max(1, limit), 5000)  # Limit between 1 and 5000
        min_confidence = max(0.0, min(1.0, min_confidence))
        
        # Get potholes from database
        potholes = db_manager.get_all_potholes(limit=limit)
        
        # Apply additional filtering
        filtered_potholes = []
        for pothole in potholes:
            # Filter by date if provided
            if since:
                try:
                    since_date = datetime.fromisoformat(since.replace('Z', '+00:00'))
                    pothole_date = pothole['detected_at']
                    if isinstance(pothole_date, str):
                        pothole_date = datetime.fromisoformat(pothole_date.replace('Z', '+00:00'))
                    if pothole_date < since_date:
                        continue
                except ValueError:
                    logger.warning(f"Invalid date format in 'since' parameter: {since}")
            
            # Filter by confidence
            if pothole['confidence'] < min_confidence:
                continue
            
            # Convert datetime to ISO string for JSON serialization
            if isinstance(pothole['detected_at'], datetime):
                pothole['detected_at'] = pothole['detected_at'].isoformat()
            
            filtered_potholes.append(pothole)
        
        return jsonify({
            'success': True,
            'potholes': filtered_potholes,
            'count': len(filtered_potholes),
            'total_before_filter': len(potholes),
            'filters_applied': {
                'limit': limit,
                'since': since,
                'min_confidence': min_confidence
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching potholes: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to fetch potholes: {str(e)}',
            'potholes': []
        }), 500

@potholes_bp.route('/potholes/bounds', methods=['POST'])
def get_potholes_in_bounds():
    """
    Get potholes within specified geographic bounds
    
    Expected JSON:
    {
        "north": latitude,
        "south": latitude,
        "east": longitude,
        "west": longitude
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['north', 'south', 'east', 'west']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        north = float(data['north'])
        south = float(data['south'])
        east = float(data['east'])
        west = float(data['west'])
        
        # Validate bounds
        if north <= south:
            return jsonify({
                'success': False,
                'error': 'North must be greater than south'
            }), 400
        
        if east <= west:
            return jsonify({
                'success': False,
                'error': 'East must be greater than west'
            }), 400
        
        # Get potholes within bounds
        potholes = db_manager.get_potholes_in_bounds(north, south, east, west)
        
        # Convert datetime objects for JSON serialization
        for pothole in potholes:
            if isinstance(pothole['detected_at'], datetime):
                pothole['detected_at'] = pothole['detected_at'].isoformat()
        
        return jsonify({
            'success': True,
            'potholes': potholes,
            'count': len(potholes),
            'bounds': {
                'north': north,
                'south': south,
                'east': east,
                'west': west
            }
        }), 200
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid coordinate values: {str(e)}'
        }), 400
    except Exception as e:
        logger.error(f"Error fetching potholes in bounds: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to fetch potholes in bounds: {str(e)}',
            'potholes': []
        }), 500

@potholes_bp.route('/potholes/density', methods=['GET'])
def get_pothole_density():
    """
    Get pothole density data for heatmap generation
    
    Query parameters:
    - grid_size: Grid cell size in degrees (default: 0.01)
    """
    try:
        grid_size = request.args.get('grid_size', default=0.01, type=float)
        grid_size = max(0.001, min(0.1, grid_size))  # Limit between 0.001 and 0.1
        
        density_data = db_manager.get_pothole_density(grid_size=grid_size)
        
        return jsonify({
            'success': True,
            'density_data': density_data,
            'grid_size': grid_size,
            'total_cells': len(density_data)
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching pothole density: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to fetch pothole density: {str(e)}',
            'density_data': []
        }), 500

@potholes_bp.route('/potholes/<int:pothole_id>', methods=['GET'])
def get_pothole_by_id(pothole_id):
    """
    Get a specific pothole by ID
    """
    try:
        # This would require a new method in db_manager
        # For now, we'll get all and filter (not efficient for production)
        potholes = db_manager.get_all_potholes(limit=5000)
        pothole = next((p for p in potholes if p['id'] == pothole_id), None)
        
        if not pothole:
            return jsonify({
                'success': False,
                'error': 'Pothole not found'
            }), 404
        
        # Convert datetime for JSON serialization
        if isinstance(pothole['detected_at'], datetime):
            pothole['detected_at'] = pothole['detected_at'].isoformat()
        
        return jsonify({
            'success': True,
            'pothole': pothole
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching pothole {pothole_id}: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to fetch pothole: {str(e)}'
        }), 500

@potholes_bp.route('/potholes/stats', methods=['GET'])
def get_pothole_statistics():
    """
    Get various pothole statistics
    """
    try:
        # Get all potholes for statistics calculation
        potholes = db_manager.get_all_potholes(limit=10000)
        
        if not potholes:
            return jsonify({
                'success': True,
                'stats': {
                    'total_count': 0,
                    'avg_confidence': 0,
                    'confidence_distribution': {},
                    'detection_trend': [],
                    'severity_distribution': {}
                }
            }), 200
        
        # Calculate statistics
        total_count = len(potholes)
        confidences = [p['confidence'] for p in potholes]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Confidence distribution
        confidence_ranges = {
            '0.0-0.3': 0,
            '0.3-0.5': 0,
            '0.5-0.7': 0,
            '0.7-0.9': 0,
            '0.9-1.0': 0
        }
        
        for conf in confidences:
            if conf < 0.3:
                confidence_ranges['0.0-0.3'] += 1
            elif conf < 0.5:
                confidence_ranges['0.3-0.5'] += 1
            elif conf < 0.7:
                confidence_ranges['0.5-0.7'] += 1
            elif conf < 0.9:
                confidence_ranges['0.7-0.9'] += 1
            else:
                confidence_ranges['0.9-1.0'] += 1
        
        # Detection trend (last 7 days)
        detection_trend = []
        today = datetime.now()
        for i in range(7):
            date = today - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            count = sum(1 for p in potholes 
                       if isinstance(p['detected_at'], (str, datetime)) and 
                       str(p['detected_at']).startswith(date_str))
            detection_trend.append({
                'date': date_str,
                'count': count
            })
        
        detection_trend.reverse()
        
        # Severity distribution based on depth
        severity_distribution = {
            'minor': 0,
            'moderate': 0,
            'severe': 0,
            'unknown': 0
        }
        
        for pothole in potholes:
            depth = pothole.get('depth')
            if depth is None:
                severity_distribution['unknown'] += 1
            elif depth > 0.15:
                severity_distribution['severe'] += 1
            elif depth > 0.08:
                severity_distribution['moderate'] += 1
            else:
                severity_distribution['minor'] += 1
        
        return jsonify({
            'success': True,
            'stats': {
                'total_count': total_count,
                'avg_confidence': round(avg_confidence, 3),
                'confidence_distribution': confidence_ranges,
                'detection_trend': detection_trend,
                'severity_distribution': severity_distribution,
                'generated_at': datetime.now().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error calculating pothole statistics: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to calculate statistics: {str(e)}'
        }), 500