import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.connection_pool = None
        self.use_storage = os.getenv("USE_SIMPLE_DB", "1") == "1"
        self.storage_file = os.getenv("SIMPLE_DB_FILE", "potholes_data.json")
        self.surveys_file = os.getenv("SIMPLE_SURVEYS_FILE", "surveys_data.json")
        self.potholes = []
        self.surveys = []
        self._next_id = 1

        if self.use_storage:
            logger.info("Using simple JSON storage for database operations.")
            self._load_storage()
        else:
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
            logger.warning("Falling back to simple JSON storage for database operations.")
            self.use_storage = True
            self._load_storage()

    def _load_storage(self):
        """Load pothole and survey data from JSON storage."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.potholes = data.get("potholes", [])
                    for pothole in self.potholes:
                        if isinstance(pothole.get("detected_at"), str):
                            try:
                                pothole["detected_at"] = datetime.fromisoformat(pothole["detected_at"])
                            except ValueError:
                                pothole["detected_at"] = datetime.utcnow()
                    if self.potholes:
                        self._next_id = max(p.get("id", 0) for p in self.potholes) + 1
            if os.path.exists(self.surveys_file):
                with open(self.surveys_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.surveys = data.get("surveys", [])
            if not self.surveys:
                self._seed_surveys()
                self._save_surveys()
            if not self.potholes:
                self._seed_sample_potholes()
                self._save_storage()
            logger.info(f"Loaded {len(self.potholes)} potholes from JSON storage.")
        except Exception as e:
            logger.error(f"Failed to load JSON storage: {e}")
            self.potholes = []
            self.surveys = []

    def _save_storage(self):
        """Persist pothole data to JSON storage."""
        if not self.use_storage:
            return
        try:
            serialised = []
            for pothole in self.potholes:
                serialised.append({
                    **pothole,
                    "detected_at": pothole.get("detected_at").isoformat() if isinstance(pothole.get("detected_at"), datetime) else pothole.get("detected_at")
                })
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump({"potholes": serialised}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save pothole storage: {e}")

    def _save_surveys(self):
        if not self.use_storage:
            return
        try:
            with open(self.surveys_file, "w", encoding="utf-8") as f:
                json.dump({"surveys": self.surveys}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save survey storage: {e}")

    def _seed_surveys(self):
        self.surveys = [
            {
                "id": 1,
                "name": "Main Street Survey",
                "description": "Downtown area inspection",
                "latitude": 12.2958,
                "longitude": 76.6394,
                "status": "active",
                "start_time": datetime.utcnow().isoformat()
            },
            {
                "id": 2,
                "name": "Highway 275 Survey",
                "description": "Highway condition assessment",
                "latitude": 12.3047,
                "longitude": 76.6412,
                "status": "active",
                "start_time": datetime.utcnow().isoformat()
            },
            {
                "id": 3,
                "name": "University Area Survey",
                "description": "Campus vicinity inspection",
                "latitude": 12.2856,
                "longitude": 76.6298,
                "status": "completed",
                "start_time": datetime.utcnow().isoformat()
            },
        ]
    
    def _seed_sample_potholes(self):
        samples = [
            {
                "pothole_id": "PH_MAIN_STREET_001",
                "survey_name": "Main Street Survey",
                "latitude": 12.2958,
                "longitude": 76.6394,
                "confidence": 0.85,
                "depth": 0.12,
                "severity": "severe",
                "area_description": "Near Main Street intersection",
            },
            {
                "pothole_id": "PH_MAIN_STREET_002",
                "survey_name": "Main Street Survey",
                "latitude": 12.2965,
                "longitude": 76.6400,
                "confidence": 0.72,
                "depth": 0.08,
                "severity": "moderate",
                "area_description": "Opposite the library",
            },
            {
                "pothole_id": "PH_HIGHWAY_275_001",
                "survey_name": "Highway 275 Survey",
                "latitude": 12.3047,
                "longitude": 76.6412,
                "confidence": 0.91,
                "depth": 0.15,
                "severity": "severe",
                "area_description": "Highway 275 northbound lane",
            },
            {
                "pothole_id": "PH_UNIVERSITY_001",
                "survey_name": "University Area Survey",
                "latitude": 12.2856,
                "longitude": 76.6298,
                "confidence": 0.68,
                "depth": 0.06,
                "severity": "minor",
                "area_description": "Near campus gate",
            },
        ]
        now = datetime.utcnow()
        for sample in samples:
            record = {
                "id": self._next_id,
                "detected_at": now,
                "status": "detected",
                "priority": 3,
                **sample,
            }
            self._next_id += 1
            self.potholes.append(record)

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        if self.use_storage:
            raise RuntimeError("Database connections are not available in storage mode.")
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
        if self.use_storage:
            return
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
        if self.use_storage:
            severity = pothole_data.get('severity')
            confidence = pothole_data.get('confidence', 0)
            depth = pothole_data.get('depth', 0) or 0
            if not severity:
                if confidence > 0.8 or depth > 0.15:
                    severity = 'severe'
                elif confidence > 0.6 or depth > 0.08:
                    severity = 'moderate'
                else:
                    severity = 'minor'
            bbox_area = None
            if pothole_data.get('bbox_width') and pothole_data.get('bbox_height'):
                bbox_area = pothole_data['bbox_width'] * pothole_data['bbox_height']

            record = {
                "id": self._next_id,
                "pothole_id": pothole_data['pothole_id'],
                "survey_name": pothole_data['survey_name'],
                "latitude": pothole_data['latitude'],
                "longitude": pothole_data['longitude'],
                "confidence": confidence,
                "depth": depth,
                "severity": severity,
                "bbox_x": pothole_data.get('bbox_x'),
                "bbox_y": pothole_data.get('bbox_y'),
                "bbox_width": pothole_data.get('bbox_width'),
                "bbox_height": pothole_data.get('bbox_height'),
                "bbox_area": bbox_area,
                "image_url": pothole_data.get('image_url'),
                "area_description": pothole_data.get('area_description'),
                "detection_type": pothole_data.get('detection_type', 'capture'),
                "device_info": pothole_data.get('device_info'),
                "status": pothole_data.get('status', 'detected'),
                "priority": pothole_data.get('priority', 3),
                "detected_at": datetime.utcnow(),
            }
            self.potholes.append(record)
            self._next_id += 1
            self._save_storage()
            return {"id": record["id"], "detected_at": record["detected_at"]}

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
        if self.use_storage:
            results = self.potholes
            if survey_name:
                results = [p for p in results if p.get("survey_name") == survey_name]
            results = sorted(results, key=lambda x: x.get("detected_at", datetime.utcnow()), reverse=True)
            limited = results[:limit]
            serialised = []
            for item in limited:
                serialised.append({
                    **item,
                    "detected_at": item.get("detected_at").isoformat() if isinstance(item.get("detected_at"), datetime) else item.get("detected_at")
                })
            return serialised

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
        if self.use_storage:
            stats = defaultdict(lambda: {
                "survey_name": None,
                "total_potholes": 0,
                "severe_count": 0,
                "moderate_count": 0,
                "minor_count": 0,
                "avg_confidence": 0,
                "avg_depth": 0,
                "first_detection": None,
                "last_detection": None,
                "live_detections": 0,
                "capture_detections": 0,
            })
            for pothole in self.potholes:
                name = pothole.get("survey_name", "Unknown")
                info = stats[name]
                info["survey_name"] = name
                info["total_potholes"] += 1
                severity = pothole.get("severity")
                if severity == "severe":
                    info["severe_count"] += 1
                elif severity == "moderate":
                    info["moderate_count"] += 1
                elif severity == "minor":
                    info["minor_count"] += 1
                info["avg_confidence"] += pothole.get("confidence", 0)
                info["avg_depth"] += pothole.get("depth", 0) or 0
                detected_at = pothole.get("detected_at")
                if isinstance(detected_at, str):
                    try:
                        detected_at = datetime.fromisoformat(detected_at)
                    except ValueError:
                        detected_at = datetime.utcnow()
                if info["first_detection"] is None or detected_at < info["first_detection"]:
                    info["first_detection"] = detected_at
                if info["last_detection"] is None or detected_at > info["last_detection"]:
                    info["last_detection"] = detected_at
                if pothole.get("detection_type") == "live":
                    info["live_detections"] += 1
                else:
                    info["capture_detections"] += 1
            results = []
            for data in stats.values():
                count = data["total_potholes"] or 1
                data["avg_confidence"] = data["avg_confidence"] / count
                data["avg_depth"] = data["avg_depth"] / count
                data["first_detection"] = data["first_detection"].isoformat() if data["first_detection"] else None
                data["last_detection"] = data["last_detection"].isoformat() if data["last_detection"] else None
                results.append(data)
            results.sort(key=lambda x: x["total_potholes"], reverse=True)
            return results

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
        if self.use_storage:
            results = []
            for pothole in self.potholes:
                lat = pothole.get("latitude")
                lng = pothole.get("longitude")
                if lat is None or lng is None:
                    continue
                if not (south <= lat <= north and west <= lng <= east):
                    continue
                if survey_name and pothole.get("survey_name") != survey_name:
                    continue
                record = pothole.copy()
                detected_at = record.get("detected_at")
                if isinstance(detected_at, datetime):
                    record["detected_at"] = detected_at.isoformat()
                results.append(record)
            results.sort(key=lambda x: x.get("detected_at"), reverse=True)
            return results

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
        if self.use_storage:
            for pothole in self.potholes:
                if pothole.get("pothole_id") == pothole_id:
                    record = pothole.copy()
                    detected_at = record.get("detected_at")
                    if isinstance(detected_at, datetime):
                        record["detected_at"] = detected_at.isoformat()
                    return record
            return None

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
        if self.use_storage:
            for pothole in self.potholes:
                if pothole.get("pothole_id") == pothole_id:
                    pothole["status"] = status
                    if notes:
                        pothole["verification_notes"] = notes
                    pothole["updated_at"] = datetime.utcnow().isoformat()
                    self._save_storage()
                    return {"id": pothole.get("id"), "status": status}
            logger.warning(f"Pothole {pothole_id} not found for status update")
            return None

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
        if self.use_storage:
            survey = {
                "id": max([s.get("id", 0) for s in self.surveys] + [0]) + 1,
                "name": survey_data['name'],
                "description": survey_data.get('description'),
                "latitude": survey_data.get('latitude'),
                "longitude": survey_data.get('longitude'),
                "surveyor_name": survey_data.get('surveyor_name'),
                "survey_type": survey_data.get('survey_type', 'routine'),
                "notes": survey_data.get('notes'),
                "status": "active",
                "start_time": datetime.utcnow().isoformat()
            }
            self.surveys.append(survey)
            self._save_surveys()
            logger.info(f"Survey created: {survey['name']}")
            return {"id": survey["id"], "name": survey["name"], "start_time": survey["start_time"]}

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
        if self.use_storage:
            stats_map = self.get_survey_statistics()
            stats_by_name = {item["survey_name"]: item for item in stats_map}
            surveys = []
            for survey in self.surveys:
                info = stats_by_name.get(survey["name"], {})
                surveys.append({
                    **survey,
                    "total_potholes": info.get("total_potholes", 0),
                    "severe_count": info.get("severe_count", 0),
                    "moderate_count": info.get("moderate_count", 0),
                    "minor_count": info.get("minor_count", 0),
                })
            surveys.sort(key=lambda x: x.get("start_time"), reverse=True)
            return surveys

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
        if self.use_storage:
            density = defaultdict(lambda: {
                "lat_grid": None,
                "lng_grid": None,
                "count": 0,
                "avg_confidence": 0,
                "surveys": set(),
            })
            for pothole in self.potholes:
                if survey_name and pothole.get("survey_name") != survey_name:
                    continue
                lat = pothole.get("latitude")
                lng = pothole.get("longitude")
                if lat is None or lng is None:
                    continue
                lat_grid = round(lat / grid_size) * grid_size
                lng_grid = round(lng / grid_size) * grid_size
                key = (lat_grid, lng_grid)
                info = density[key]
                info["lat_grid"] = lat_grid
                info["lng_grid"] = lng_grid
                info["count"] += 1
                info["avg_confidence"] += pothole.get("confidence", 0)
                info["surveys"].add(pothole.get("survey_name", "Unknown"))
            results = []
            for info in density.values():
                if info["count"] == 0:
                    continue
                results.append({
                    "lat_grid": info["lat_grid"],
                    "lng_grid": info["lng_grid"],
                    "count": info["count"],
                    "avg_confidence": info["avg_confidence"] / info["count"],
                    "surveys": ", ".join(sorted(info["surveys"]))
                })
            results.sort(key=lambda x: x["count"], reverse=True)
            return results

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
        if self.use_storage:
            cutoff = datetime.utcnow() - timedelta(days=days)
            filtered = []
            for pothole in self.potholes:
                detected_at = pothole.get("detected_at")
                if isinstance(detected_at, str):
                    try:
                        detected_at = datetime.fromisoformat(detected_at)
                    except ValueError:
                        detected_at = datetime.utcnow()
                if detected_at < cutoff:
                    continue
                if survey_name and pothole.get("survey_name") != survey_name:
                    continue
                filtered.append((pothole, detected_at))

            daily = defaultdict(lambda: {"detection_date": None, "daily_count": 0, "daily_avg_confidence": 0})
            hourly = defaultdict(lambda: {"hour": None, "hourly_count": 0})
            severity_stats = defaultdict(lambda: {"severity": None, "count": 0, "avg_confidence": 0, "avg_depth": 0})

            for pothole, detected_at in filtered:
                date_key = detected_at.date().isoformat()
                info = daily[date_key]
                info["detection_date"] = date_key
                info["daily_count"] += 1
                info["daily_avg_confidence"] += pothole.get("confidence", 0)

                hour = detected_at.hour
                hinfo = hourly[hour]
                hinfo["hour"] = hour
                hinfo["hourly_count"] += 1

                severity = pothole.get("severity", "unknown")
                sinfo = severity_stats[severity]
                sinfo["severity"] = severity
                sinfo["count"] += 1
                sinfo["avg_confidence"] += pothole.get("confidence", 0)
                sinfo["avg_depth"] += pothole.get("depth", 0) or 0

            daily_data = []
            for item in daily.values():
                count = item["daily_count"] or 1
                item["daily_avg_confidence"] = item["daily_avg_confidence"] / count
                daily_data.append(item)
            daily_data.sort(key=lambda x: x["detection_date"], reverse=True)

            hourly_data = list(hourly.values())
            hourly_data.sort(key=lambda x: x["hour"])

            severity_data = []
            for item in severity_stats.values():
                count = item["count"] or 1
                item["avg_confidence"] = item["avg_confidence"] / count
                item["avg_depth"] = item["avg_depth"] / count
                severity_data.append(item)

            return {
                "daily": daily_data,
                "hourly": hourly_data,
                "severity": severity_data,
            }

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
        if self.use_storage:
            return
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Database connections closed")

# Global database manager instance
db_manager = DatabaseManager()