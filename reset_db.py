#!/usr/bin/env python3
"""
Reset database script - cleans up any existing tables and recreates them
Run this if you have database schema conflicts
"""

import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def reset_database():
    """Reset the database by dropping and recreating tables"""
    try:
        # Database configuration
        db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password')
        }
        db_name = os.getenv('DB_NAME', 'pothole_db')
        
        print("🗑️  Resetting database...")
        
        # Connect to our database
        conn = psycopg2.connect(database=db_name, **db_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Drop existing tables
        print("Dropping existing tables...")
        cursor.execute("DROP TABLE IF EXISTS potholes CASCADE;")
        
        # Enable PostGIS
        print("Enabling PostGIS extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        
        # Create fresh table
        print("Creating fresh potholes table...")
        cursor.execute("""
            CREATE TABLE potholes (
                id SERIAL PRIMARY KEY,
                latitude DOUBLE PRECISION NOT NULL,
                longitude DOUBLE PRECISION NOT NULL,
                location GEOMETRY(POINT, 4326),
                confidence REAL NOT NULL,
                depth REAL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_width INTEGER,
                bbox_height INTEGER,
                image_url TEXT,
                detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create indexes
        print("Creating indexes...")
        cursor.execute("CREATE INDEX idx_potholes_location ON potholes USING GIST (location);")
        cursor.execute("CREATE INDEX idx_potholes_detected_at ON potholes (detected_at DESC);")
        cursor.execute("CREATE INDEX idx_potholes_confidence ON potholes (confidence DESC);")
        
        # Insert sample data
        
        # Test the setup
        cursor.execute("SELECT COUNT(*) FROM potholes;")
        count = cursor.fetchone()[0]
        print(f"✅ Verification: Found {count} pothole records in database.")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Database reset failed: {e}")
        return False

if __name__ == "__main__":
    if reset_database():
        print("\n🚀 You can now run 'python app.py' to start the application!")
    else:
        print("\n❌ Database reset failed. Please check your configuration.")