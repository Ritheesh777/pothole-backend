import os
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
import logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.connection_pool = None
        self.init_connection_pool()
    
    def init_connection_pool(self):
        """Initialize connection pool"""
        try:
            # Database configuration
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME', 'pothole_db'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'password')
            }
            
            self.connection_pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                **db_config
            )
            logger.info("Database connection pool initialized successfully")
            
            # Create tables if they don't exist
            self.create_tables()
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def create_tables(self):
        """Create necessary tables with all required fields"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Enable PostGIS extension
                    logger.info("Enabling PostGIS extension...")
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
                    
                    # Drop existing table to recreate with new schema
                    logger.info("Updating database schema...")
                    cursor.execute("DROP TABLE IF EXISTS potholes CASCADE;")
                    
                    # Create enhanced potholes table
                    logger.info("Creating enhanced potholes table...")
                    create_table_sql = """
                    CREATE TABLE potholes (
                        id SERIAL PRIMARY KEY,
                        
                        -- Unique Pothole Identification
                        pothole_id VARCHAR(100) UNIQUE NOT NULL,
                        
                        -- Survey Information
                        survey_name VARCHAR(255) NOT NULL,
                        survey_area VARCHAR(255),
                        detection_type VARCHAR(50) DEFAULT 'capture', -- 'live' or 'capture'
                        
                        -- Location Data
                        latitude DOUBLE PRECISION NOT NULL,
                        longitude DOUBLE PRECISION NOT NULL,
                        location GEOMETRY(POINT, 4326),
                        area_description TEXT,
                        
                        -- Detection Metrics
                        confidence REAL NOT NULL,
                        depth REAL,
                        severity VARCHAR(20), -- 'minor', 'moderate', 'severe'
                        
                        -- Bounding Box Data
                        bbox_x INTEGER,
                        bbox_y INTEGER,
                        bbox_width INTEGER,
                        bbox_height INTEGER,
                        bbox_area INTEGER,
                        
                        -- Image and Media
                        image_url TEXT,
                        image_base64 TEXT, -- For storing small thumbnails
                        
                        -- Timestamps
                        detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Additional Metadata
                        device_info JSONB, -- Store device/camera info
                        weather_conditions VARCHAR(100),
                        road_type VARCHAR(100),
                        traffic_conditions VARCHAR(100),
                        
                        -- Quality and Verification
                        manual_verified BOOLEAN DEFAULT FALSE,
                        verification_notes TEXT,
                        quality_score REAL, -- Overall detection quality score
                        
                        -- Status and Priority
                        status VARCHAR(50) DEFAULT 'detected', -- 'detected', 'reported', 'in_repair', 'fixed'
                        priority INTEGER DEFAULT 3, -- 1=low, 2=medium, 3=high, 4=critical
                        
                        -- Repair Tracking
                        repair_cost_estimate DECIMAL(10,2),
                        repair_scheduled_date TIMESTAMP WITH TIME ZONE,
                        repair_completed_date TIMESTAMP WITH TIME ZONE,
                        repair_notes TEXT
                    );
                    """
                    cursor.execute(create_table_sql)
                    
                    # Create surveys table for survey management
                    logger.info("Creating surveys table...")
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS surveys (
                            id SERIAL PRIMARY KEY,
                            name VARCHAR(255) UNIQUE NOT NULL,
                            description TEXT,
                            start_location GEOMETRY(POINT, 4326),
                            area_bounds GEOMETRY(POLYGON, 4326),
                            start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            end_time TIMESTAMP WITH TIME ZONE,
                            status VARCHAR(50) DEFAULT 'active', -- 'active', 'completed', 'paused'
                            total_potholes INTEGER DEFAULT 0,
                            surveyor_name VARCHAR(255),
                            survey_type VARCHAR(100), -- 'routine', 'emergency', 'follow_up'
                            notes TEXT,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    # Create indexes for better performance
                    logger.info("Creating indexes...")
                    indexes = [
                        "CREATE INDEX IF NOT EXISTS idx_potholes_pothole_id ON potholes (pothole_id);",
                        "CREATE INDEX IF NOT EXISTS idx_potholes_survey_name ON potholes (survey_name);",
                        "CREATE INDEX IF NOT EXISTS idx_potholes_location ON potholes USING GIST (location);",
                        "CREATE INDEX IF NOT EXISTS idx_potholes_detected_at ON potholes (detected_at DESC);",
                        "CREATE INDEX IF NOT EXISTS idx_potholes_confidence ON potholes (confidence DESC);",
                        "CREATE INDEX IF NOT EXISTS idx_potholes_severity ON potholes (severity);",
                        "CREATE INDEX IF NOT EXISTS idx_potholes_status ON potholes (status);",
                        "CREATE INDEX IF NOT EXISTS idx_potholes_priority ON potholes (priority DESC);",
                        "CREATE INDEX IF NOT EXISTS idx_surveys_name ON surveys (name);",
                        "CREATE INDEX IF NOT EXISTS idx_surveys_status ON surveys (status);",
                        "CREATE INDEX IF NOT EXISTS idx_surveys_start_time ON surveys (start_time DESC);"
                    ]
                    
                    for index_sql in indexes:
                        cursor.execute(index_sql)
                    
                    # Create trigger to update location column
                    cursor.execute("""
                        CREATE OR REPLACE FUNCTION update_location()
                        RETURNS TRIGGER AS $$
                        BEGIN
                            NEW.location = ST_SetSRID(ST_MakePoint(NEW.longitude, NEW.latitude), 4326);
                            NEW.updated_at = CURRENT_TIMESTAMP;
                            RETURN NEW;
                        END;
                        $$ LANGUAGE plpgsql;
                        
                        CREATE TRIGGER trigger_update_location
                            BEFORE INSERT OR UPDATE ON potholes
                            FOR EACH ROW
                            EXECUTE FUNCTION update_location();
                    """)
                    
                    # Insert sample data for testing
                    cursor.execute("SELECT COUNT(*) FROM potholes;")
                    count = cursor.fetchone()[0]
                    
                    if count == 0:
                        logger.info("Adding sample data...")
                        sample_surveys = [
                            ("Main Street Survey", "Downtown area inspection", 12.2958, 76.6394),
                            ("Highway 275 Survey", "Highway condition assessment", 12.3047, 76.6412),
                            ("University Area Survey", "Campus vicinity inspection", 12.2856, 76.6298)
                        ]
                        
                        for survey_name, description, lat, lng in sample_surveys:
                            cursor.execute("""
                                INSERT INTO surveys (name, description, start_location)
                                VALUES (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))
                                ON CONFLICT (name) DO NOTHING
                            """, (survey_name, description, lng, lat))
                        
                        sample_potholes = [
                            ("PH_MAIN_STREET_001", "Main Street Survey", 12.2958, 76.6394, 0.85, 0.12, "severe"),
                            ("PH_MAIN_STREET_002", "Main Street Survey", 12.2965, 76.6400, 0.72, 0.08, "moderate"),
                            ("PH_HIGHWAY_275_001", "Highway 275 Survey", 12.3047, 76.6412, 0.91, 0.15, "severe"),
                            ("PH_UNIVERSITY_001", "University Area Survey", 12.2856, 76.6298, 0.68, 0.06, "minor")
                        ]
                        
                        for pothole_id, survey_name, lat, lng, conf, depth, severity in sample_potholes:
                            cursor.execute("""
                                INSERT INTO potholes 
                                (pothole_id, survey_name, latitude, longitude, confidence, depth, severity, area_description)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """, (pothole_id, survey_name, lat, lng, conf, depth, severity, f"Area around {lat:.4f}, {lng:.4f}"))
                        
                        logger.info("Added sample surveys and potholes")
                    
                    conn.commit()
            logger.info("Database schema updated successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def insert_pothole(self, pothole_data):
        """Insert a new pothole record with all fields"""
        insert_sql = """
        INSERT INTO potholes 
        (pothole_id, survey_name, latitude, longitude, confidence, depth, severity,
         bbox_x, bbox_y, bbox_width, bbox_height, bbox_area, image_url, 
         area_description, detection_type, device_info)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id, detected_at;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Calculate severity if not provided
                    severity = pothole_data.get('severity')
                    if not severity:
                        confidence = pothole_data['confidence']
                        depth = pothole_data.get('depth', 0)
                        if confidence > 0.8 or depth > 0.15:
                            severity = 'severe'
                        elif confidence > 0.6 or depth > 0.08:
                            severity = 'moderate'
                        else:
                            severity = 'minor'
                    
                    # Calculate bounding box area
                    bbox_area = None
                    if pothole_data.get('bbox_width') and pothole_data.get('bbox_height'):
                        bbox_area = pothole_data['bbox_width'] * pothole_data['bbox_height']
                    
                    cursor.execute(insert_sql, (
                        pothole_data['pothole_id'],
                        pothole_data['survey_name'],
                        pothole_data['latitude'],
                        pothole_data['longitude'],
                        pothole_data['confidence'],
                        pothole_data.get('depth'),
                        severity,
                        pothole_data.get('bbox_x'),
                        pothole_data.get('bbox_y'),
                        pothole_data.get('bbox_width'),
                        pothole_data.get('bbox_height'),
                        bbox_area,
                        pothole_data.get('image_url'),
                        pothole_data.get('area_description'),
                        pothole_data.get('detection_type', 'capture'),
                        pothole_data.get('device_info')
                    ))
                    result = cursor.fetchone()
                    conn.commit()
                    logger.info(f"Pothole inserted with ID: {result['id']}")
                    return dict(result)
        except Exception as e:
            logger.error(f"Failed to insert pothole: {e}")
            raise
    
    def get_all_potholes(self, limit=1000, survey_name=None):
        """Get all potholes with optional survey filter"""
        base_sql = """
        SELECT id, pothole_id, survey_name, latitude, longitude, confidence, depth, severity,
               bbox_x, bbox_y, bbox_width, bbox_height, image_url, area_description,
               detection_type, detected_at, status, priority
        FROM potholes
        """
        
        if survey_name:
            select_sql = base_sql + " WHERE survey_name = %s ORDER BY detected_at DESC LIMIT %s;"
            params = (survey_name, limit)
        else:
            select_sql = base_sql + " ORDER BY detected_at DESC LIMIT %s;"
            params = (limit,)
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(select_sql, params)
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to fetch potholes: {e}")
            raise
    
    def get_potholes_by_survey(self, survey_name):
        """Get all potholes for a specific survey"""
        return self.get_all_potholes(survey_name=survey_name)
    
    def get_survey_statistics(self):
        """Get comprehensive survey statistics"""
        stats_sql = """
        SELECT 
            survey_name,
            COUNT(*) as total_potholes,
            COUNT(CASE WHEN severity = 'severe' THEN 1 END) as severe_count,
            COUNT(CASE WHEN severity = 'moderate' THEN 1 END) as moderate_count,
            COUNT(CASE WHEN severity = 'minor' THEN 1 END) as minor_count,
            AVG(confidence) as avg_confidence,
            AVG(depth) as avg_depth,
            MIN(detected_at) as first_detection,
            MAX(detected_at) as last_detection,
            COUNT(CASE WHEN detection_type = 'live' THEN 1 END) as live_detections,
            COUNT(CASE WHEN detection_type = 'capture' THEN 1 END) as capture_detections
        FROM potholes
        GROUP BY survey_name
        ORDER BY total_potholes DESC;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(stats_sql)
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to fetch survey statistics: {e}")
            raise
    
    def get_potholes_in_bounds(self, north, south, east, west, survey_name=None):
        """Get potholes within geographic bounds"""
        base_sql = """
        SELECT id, pothole_id, survey_name, latitude, longitude, confidence, depth, severity,
               bbox_x, bbox_y, bbox_width, bbox_height, image_url, area_description,
               detection_type, detected_at, status, priority
        FROM potholes
        WHERE location && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
        AND latitude BETWEEN %s AND %s
        AND longitude BETWEEN %s AND %s
        """
        
        if survey_name:
            bounds_sql = base_sql + " AND survey_name = %s ORDER BY detected_at DESC;"
            params = (west, south, east, north, south, north, west, east, survey_name)
        else:
            bounds_sql = base_sql + " ORDER BY detected_at DESC;"
            params = (west, south, east, north, south, north, west, east)
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(bounds_sql, params)
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to fetch potholes in bounds: {e}")
            raise
    
    def get_pothole_by_id(self, pothole_id):
        """Get a specific pothole by its unique ID"""
        select_sql = """
        SELECT * FROM potholes WHERE pothole_id = %s;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(select_sql, (pothole_id,))
                    result = cursor.fetchone()
                    return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to fetch pothole {pothole_id}: {e}")
            raise
    
    def update_pothole_status(self, pothole_id, status, notes=None):
        """Update pothole status and add notes"""
        update_sql = """
        UPDATE potholes 
        SET status = %s, verification_notes = COALESCE(%s, verification_notes), updated_at = CURRENT_TIMESTAMP
        WHERE pothole_id = %s
        RETURNING id, status;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(update_sql, (status, notes, pothole_id))
                    result = cursor.fetchone()
                    conn.commit()
                    if result:
                        logger.info(f"Updated pothole {pothole_id} status to {status}")
                        return dict(result)
                    else:
                        logger.warning(f"Pothole {pothole_id} not found for status update")
                        return None
        except Exception as e:
            logger.error(f"Failed to update pothole status: {e}")
            raise
    
    def create_survey(self, survey_data):
        """Create a new survey"""
        insert_sql = """
        INSERT INTO surveys (name, description, start_location, surveyor_name, survey_type, notes)
        VALUES (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s, %s)
        RETURNING id, name, start_time;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(insert_sql, (
                        survey_data['name'],
                        survey_data.get('description'),
                        survey_data.get('longitude'),
                        survey_data.get('latitude'),
                        survey_data.get('surveyor_name'),
                        survey_data.get('survey_type', 'routine'),
                        survey_data.get('notes')
                    ))
                    result = cursor.fetchone()
                    conn.commit()
                    logger.info(f"Survey created: {result['name']}")
                    return dict(result)
        except Exception as e:
            logger.error(f"Failed to create survey: {e}")
            raise
    
    def get_all_surveys(self):
        """Get all surveys with their statistics"""
        surveys_sql = """
        SELECT s.*, 
               COALESCE(p.total_potholes, 0) as total_potholes,
               COALESCE(p.severe_count, 0) as severe_count,
               COALESCE(p.moderate_count, 0) as moderate_count,
               COALESCE(p.minor_count, 0) as minor_count
        FROM surveys s
        LEFT JOIN (
            SELECT survey_name,
                   COUNT(*) as total_potholes,
                   COUNT(CASE WHEN severity = 'severe' THEN 1 END) as severe_count,
                   COUNT(CASE WHEN severity = 'moderate' THEN 1 END) as moderate_count,
                   COUNT(CASE WHEN severity = 'minor' THEN 1 END) as minor_count
            FROM potholes
            GROUP BY survey_name
        ) p ON s.name = p.survey_name
        ORDER BY s.start_time DESC;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(surveys_sql)
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to fetch surveys: {e}")
            raise
    
    def get_pothole_density(self, grid_size=0.01, survey_name=None):
        """Get pothole density data for heatmap"""
        base_sql = """
        SELECT 
            ROUND(latitude / %s) * %s as lat_grid,
            ROUND(longitude / %s) * %s as lng_grid,
            COUNT(*) as count,
            AVG(confidence) as avg_confidence,
            string_agg(DISTINCT survey_name, ', ') as surveys
        FROM potholes
        """
        
        if survey_name:
            density_sql = base_sql + " WHERE survey_name = %s GROUP BY lat_grid, lng_grid HAVING COUNT(*) > 0 ORDER BY count DESC;"
            params = (grid_size, grid_size, grid_size, grid_size, survey_name)
        else:
            density_sql = base_sql + " GROUP BY lat_grid, lng_grid HAVING COUNT(*) > 0 ORDER BY count DESC;"
            params = (grid_size, grid_size, grid_size, grid_size)
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(density_sql, params)
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to fetch pothole density: {e}")
            raise
    
    def get_detailed_analytics(self, days=30, survey_name=None):
        """Get detailed analytics for the dashboard"""
        base_where = f"WHERE detected_at >= CURRENT_TIMESTAMP - INTERVAL '{days} days'"
        if survey_name:
            base_where += f" AND survey_name = '{survey_name}'"
        
        analytics_sql = f"""
        WITH daily_stats AS (
            SELECT 
                DATE(detected_at) as detection_date,
                COUNT(*) as daily_count,
                AVG(confidence) as daily_avg_confidence
            FROM potholes
            {base_where}
            GROUP BY DATE(detected_at)
            ORDER BY detection_date DESC
        ),
        hourly_distribution AS (
            SELECT 
                EXTRACT(HOUR FROM detected_at) as hour,
                COUNT(*) as hourly_count
            FROM potholes
            {base_where}
            GROUP BY EXTRACT(HOUR FROM detected_at)
            ORDER BY hour
        ),
        severity_stats AS (
            SELECT 
                severity,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                AVG(depth) as avg_depth
            FROM potholes
            {base_where}
            GROUP BY severity
        )
        SELECT 
            'daily' as stat_type,
            json_agg(daily_stats) as data
        FROM daily_stats
        UNION ALL
        SELECT 
            'hourly' as stat_type,
            json_agg(hourly_distribution) as data
        FROM hourly_distribution  
        UNION ALL
        SELECT 
            'severity' as stat_type,
            json_agg(severity_stats) as data
        FROM severity_stats;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(analytics_sql)
                    results = cursor.fetchall()
                    
                    # Transform results into a more usable format
                    analytics = {}
                    for row in results:
                        analytics[row['stat_type']] = row['data']
                    
                    return analytics
        except Exception as e:
            logger.error(f"Failed to fetch detailed analytics: {e}")
            raise
    
    def close_connections(self):
        """Close all connections in the pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Database connections closed")

# Global database manager instance
db_manager = DatabaseManager()