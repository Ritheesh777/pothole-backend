from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import routes
from routes.detect import detect_bp
from routes.potholes import potholes_bp

# Import database manager for initialization
from database.db import db_manager

from routes.pothole_save import pothole_save_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Enable CORS for all routes
    CORS(app, origins="*")
    
    # Register blueprints
    app.register_blueprint(detect_bp)
    app.register_blueprint(potholes_bp)
    app.register_blueprint(pothole_save_bp)
    
    # Root route
    @app.route('/')
    def index():
        return jsonify({
            'message': 'Smart Pothole Detection API',
            'version': '1.0.0',
            'status': 'running',
            'endpoints': {
                'detection': {
                    'POST /detect': 'Detect potholes in single image',
                    'POST /detect/batch': 'Detect potholes in multiple images',
                    'GET /detect/health': 'Check detection service health'
                },
                'potholes': {
                    'GET /potholes': 'Get all potholes with optional filters',
                    'POST /potholes/bounds': 'Get potholes within geographic bounds',
                    'GET /potholes/density': 'Get pothole density data for heatmap',
                    'GET /potholes/<id>': 'Get specific pothole by ID',
                    'GET /potholes/stats': 'Get pothole statistics'
                }
            }
        })
    
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        try:
            # Test database connection
            potholes = db_manager.get_all_potholes(limit=1)
            db_healthy = True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            db_healthy = False
        
        return jsonify({
            'status': 'healthy' if db_healthy else 'degraded',
            'components': {
                'database': db_healthy,
                'api': True
            },
            'timestamp': db_manager.__class__.__name__ if db_healthy else None
        }), 200 if db_healthy else 503
    
    # Error handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'message': str(error.description)
        }), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'error': 'Not found',
            'message': 'The requested resource was not found'
        }), 404
    
    @app.errorhandler(413)
    def payload_too_large(error):
        return jsonify({
            'success': False,
            'error': 'Payload too large',
            'message': 'File size exceeds the maximum limit of 16MB'
        }), 413
    
    @app.errorhandler(500)
    def internal_server_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500
    
    # Add this route for serving images
    @app.route('/images/<filename>')
    def serve_image(filename):
        images_dir = os.path.join(os.getcwd(), 'images')
        return send_from_directory(images_dir, filename)
    
    # Request logging middleware
    @app.before_request
    def log_request_info():
        if request.endpoint not in ['health_check', 'index']:
            logger.info(f'{request.method} {request.url} - {request.remote_addr}')
    
    @app.after_request
    def log_response_info(response):
        if request.endpoint not in ['health_check', 'index']:
            logger.info(f'{request.method} {request.url} - {response.status_code}')
        return response
    
    return app

def initialize_models():
    """Initialize ML models on startup"""
    try:
        logger.info("Initializing machine learning models...")
        
        # Import and initialize models
        from models.yolov8_model import pothole_detector
        from utils.depth_estimation import depth_estimator
        
        logger.info("Models initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        return False

# Initialize models and create the WSGI app at import time
_models_initialized = initialize_models()
if not _models_initialized:
    logger.warning("Some models failed to initialize. Continuing with limited functionality.")

app = create_app()

if __name__ == '__main__':
    # Get configuration from environment variables
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    
    logger.info(f"Starting Flask application on {host}:{port}")
    logger.info(f"Debug mode: {debug_mode}")
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug_mode,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
    finally:
        # Cleanup database connections
        try:
            db_manager.close_connections()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")