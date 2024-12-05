from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import sqlite3
import os
from pathlib import Path
import logging
import urllib.parse
import requests
from deepface import DeepFace
import uvicorn
from typing import List
import json
import shutil
from collections import defaultdict
import traceback
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
IMAGES_DIR = Path("profile_images")
IMAGES_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Face Analyzer")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and profile images
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/profile_images", StaticFiles(directory="profile_images"), name="profile_images")

def init_db():
    """Initialize database with updated schema"""
    conn = sqlite3.connect('profile_images.db')
    c = conn.cursor()
    
    # Drop existing tables if they exist
    c.execute('DROP TABLE IF EXISTS race_analysis')
    c.execute('DROP TABLE IF EXISTS images')
    
    # Create images table
    c.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            race TEXT,
            age TEXT,
            gender TEXT,
            confidence REAL
        )
    ''')
    
    # Create race_analysis table with sub-ethnicity support
    c.execute('''
        CREATE TABLE IF NOT EXISTS race_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            race TEXT NOT NULL,
            confidence REAL NOT NULL,
            is_sub_ethnicity INTEGER DEFAULT 0,
            parent_race TEXT,
            FOREIGN KEY (image_id) REFERENCES images(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def get_dominant_race_group(race_scores):
    """
    Enhanced race classification with detailed sub-ethnicity detection
    """
    # Primary race categories with detailed sub-ethnicities
    race_mapping = {
        'South Asian': {
            'sub_ethnicities': {
                'Indian': ['indian', 'dravidian', 'indo-aryan'],
                'Pakistani': ['pakistani', 'punjabi', 'sindhi'],
                'Bangladeshi': ['bangladeshi', 'bengali'],
                'Sri Lankan': ['sri lankan', 'sinhalese', 'tamil'],
                'Nepali': ['nepali', 'sherpa'],
            },
            'keywords': ['indian', 'south asian', 'pakistani', 'bangladeshi', 'sri lankan', 'nepali'],
            'weight': 1.3
        },
        'East Asian': {
            'sub_ethnicities': {
                'Chinese': ['chinese', 'han'],
                'Japanese': ['japanese'],
                'Korean': ['korean'],
            },
            'keywords': ['east asian', 'chinese', 'japanese', 'korean', 'mongolian'],
            'weight': 1.2
        },
        'Southeast Asian': {
            'sub_ethnicities': {
                'Vietnamese': ['vietnamese'],
                'Thai': ['thai'],
                'Filipino': ['filipino'],
                'Malaysian': ['malaysian', 'malay'],
                'Indonesian': ['indonesian']
            },
            'keywords': ['southeast asian', 'vietnamese', 'thai', 'filipino', 'malaysian', 'indonesian'],
            'weight': 1.2
        },
        'Middle Eastern': {
            'sub_ethnicities': {
                'Arab': ['arab', 'saudi', 'egyptian'],
                'Persian': ['persian', 'iranian'],
                'Turkish': ['turkish']
            },
            'keywords': ['middle eastern', 'arab', 'persian', 'turkish'],
            'weight': 1.1
        },
        'Hispanic/Latino': {
            'sub_ethnicities': {
                'Mexican': ['mexican'],
                'Brazilian': ['brazilian'],
                'Colombian': ['colombian'],
                'Puerto Rican': ['puerto rican']
            },
            'keywords': ['latino hispanic', 'latin american', 'mexican', 'brazilian'],
            'weight': 1.1
        },
        'Black/African': {
            'sub_ethnicities': {
                'West African': ['west african', 'nigerian', 'ghanaian'],
                'East African': ['east african', 'ethiopian', 'kenyan'],
                'African American': ['african american']
            },
            'keywords': ['black', 'african', 'african american'],
            'weight': 1.0
        }
    }

    # Initialize scores
    weighted_scores = defaultdict(float)
    sub_ethnicity_scores = defaultdict(lambda: defaultdict(float))
    original_scores = {}

    # First pass: Calculate initial scores and sub-ethnicity detection
    for race, score in race_scores.items():
        race_lower = race.lower()
        original_scores[race] = score
        
        # Apply voting for each category and sub-ethnicity
        for category, info in race_mapping.items():
            if any(keyword in race_lower for keyword in info['keywords']):
                weighted_scores[category] += score * info['weight']
                
                # Sub-ethnicity detection
                for sub_eth, patterns in info['sub_ethnicities'].items():
                    if any(pattern in race_lower for pattern in patterns):
                        sub_ethnicity_scores[category][sub_eth] += score * info['weight']

    # Handle special case for generic "asian"
    if any('asian' in race.lower() for race in race_scores.keys()):
        asian_score = race_scores.get('asian', 0)
        if asian_score > 0:
            weighted_scores['South Asian'] += asian_score * 0.4
            sub_ethnicity_scores['South Asian']['Indian'] += asian_score * 0.3
            sub_ethnicity_scores['South Asian']['Pakistani'] += asian_score * 0.1

    # Normalize scores
    total_weighted = sum(weighted_scores.values())
    if total_weighted > 0:
        for race in weighted_scores:
            weighted_scores[race] = (weighted_scores[race] / total_weighted) * 100
            
            # Normalize sub-ethnicity scores
            if sub_ethnicity_scores[race]:
                total_sub = sum(sub_ethnicity_scores[race].values())
                if total_sub > 0:
                    for sub_eth in sub_ethnicity_scores[race]:
                        sub_ethnicity_scores[race][sub_eth] = (sub_ethnicity_scores[race][sub_eth] / total_sub) * 100

    # Sort races by confidence
    sorted_races = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_races:
        return "Unidentified", 0.0, {}

    # Get primary race and sub-ethnicity
    primary_race = sorted_races[0][0]
    primary_confidence = sorted_races[0][1] / 100
    
    # Get dominant sub-ethnicity for primary race
    primary_sub_ethnicity = None
    sub_eth_scores = sub_ethnicity_scores[primary_race]
    if sub_eth_scores:
        primary_sub_ethnicity = max(sub_eth_scores.items(), key=lambda x: x[1])

    # Format detailed distribution
    detailed_scores = {
        'Primary Race': f"{primary_race} ({primary_confidence:.1%})",
        'Sub Ethnicity': f"{primary_sub_ethnicity[0]} ({primary_sub_ethnicity[1]:.1f}%)" if primary_sub_ethnicity else "Unspecified",
        'Detailed Distribution': {
            race: f"{score:.1f}%" for race, score in sorted_races if score > 1.0
        },
        'Sub Ethnicities': {
            race: {sub: f"{score:.1f}%" for sub, score in subs.items() if score > 1.0}
            for race, subs in sub_ethnicity_scores.items()
            if any(score > 1.0 for score in subs.values())
        },
        'Raw Scores': {
            race: f"{score:.1f}%" for race, score in original_scores.items()
        }
    }

    return primary_race, primary_confidence, detailed_scores

def analyze_face(image_path: str):
    """Analyze face using DeepFace with enhanced race detection"""
    try:
        # Use multiple models for better accuracy
        backends = ['opencv', 'retinaface', 'mtcnn']
        results = []
        
        for backend in backends:
            try:
                result = DeepFace.analyze(
                    img_path=image_path,
                    actions=['age', 'gender', 'race'],
                    enforce_detection=True,
                    detector_backend=backend,
                    align=True,
                    silent=True
                )
                if isinstance(result, list):
                    result = result[0]
                results.append(result)
            except Exception as e:
                print(f"Backend {backend} failed: {str(e)}")
                continue

        if not results:
            raise Exception("All detection backends failed")

        # Use the most confident result
        result = max(results, key=lambda x: max(x.get('race', {}).values()))
        
        gender = result.get('dominant_gender', result.get('gender', 'Unknown'))
        age = result.get('age', 'Unknown')
        
        race_scores = result.get('race', {})
        if race_scores:
            # Get enhanced race classification with sub-ethnicities
            dominant_race, race_confidence, detailed_scores = get_dominant_race_group(race_scores)
            
            # Debug logging
            print("\nRace Analysis Details:")
            print(f"Race scores: {race_scores}")
            print(f"Dominant race: {dominant_race}")
            print(f"Detailed scores: {json.dumps(detailed_scores, indent=2)}")
            
            return {
                "race": dominant_race,
                "confidence": race_confidence,
                "age": str(age),
                "gender": gender,
                "detailed_analysis": detailed_scores,
                "raw_scores": race_scores  # Include raw scores for debugging
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error in analyze_face: {str(e)}")
        traceback.print_exc()
        return None

def process_profile(name: str, url: str, index: int):
    """Process a single profile"""
    try:
        filename = f"{name.replace(' ', '_')}_{index}.jpg"
        image_path = download_image(url, filename)
        
        if image_path:
            logger.info(f"Processing image: {image_path}")
            analysis = analyze_face(image_path)
            
            if analysis:
                image_id = store_metadata(name, filename)
                if image_id:
                    store_analysis(image_id, analysis)
                    logger.info(f"Successfully analyzed {name}")
                    return True, analysis
                else:
                    logger.error(f"Failed to store metadata for {name}")
            else:
                logger.error(f"Face analysis failed for {name}")
        else:
            logger.error(f"Failed to download image for {name}")
        return False, None
    except Exception as e:
        logger.error(f"Error processing profile {name}: {str(e)}")
        return False, None

def get_google_drive_file_id(url: str) -> str:
    """Extract file ID from Google Drive URL"""
    if 'drive.google.com' not in url:
        return None
    
    file_id = None
    if '/file/d/' in url:
        # Handle direct file links
        start = url.find('/file/d/') + 8
        end = url.find('/', start)
        if end == -1:
            end = url.find('?', start)
        if end == -1:
            end = len(url)
        file_id = url[start:end]
    elif 'id=' in url:
        # Handle 'id=' format
        start = url.find('id=') + 3
        end = url.find('&', start)
        if end == -1:
            end = len(url)
        file_id = url[start:end]
    
    return file_id

def download_image(url: str, filename: str):
    """Download image from URL with support for various sources"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        # Check if it's a Google Drive URL
        file_id = get_google_drive_file_id(url)
        if file_id:
            # Use the direct download link format
            url = f'https://drive.google.com/uc?export=download&id={file_id}'
            headers['Cookie'] = 'download_warning=1'  # Skip the warning page

        clean_url = urllib.parse.quote(url.strip(), safe=':/?=&')
        response = requests.get(clean_url, stream=True, timeout=30, headers=headers)
        response.raise_for_status()

        # Check if it's actually an image
        content_type = response.headers.get('content-type', '').lower()
        if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
            # For Google Drive, the content-type might not be accurate
            if not file_id or 'text/html' in content_type:
                raise ValueError(f"URL does not point to an image: {content_type}")

        image_path = IMAGES_DIR / filename
        with open(image_path, 'wb') as f:
            f.write(response.content)

        # Verify the downloaded file is an image
        try:
            from PIL import Image
            img = Image.open(image_path)
            img.verify()  # Verify it's actually an image
            img.close()
            
            # Convert to JPEG if it's not
            if image_path.suffix.lower() not in ['.jpg', '.jpeg']:
                img = Image.open(image_path)
                img = img.convert('RGB')
                jpeg_path = IMAGES_DIR / f"{image_path.stem}.jpg"
                img.save(jpeg_path, 'JPEG')
                img.close()
                if image_path != jpeg_path:
                    image_path.unlink()  # Remove original file
                image_path = jpeg_path
                
            return str(image_path)
        except Exception as e:
            if image_path.exists():
                image_path.unlink()  # Remove invalid file
            raise ValueError(f"Downloaded file is not a valid image: {str(e)}")

    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        return None

def store_metadata(name: str, image_path: str):
    """Store profile metadata in database"""
    try:
        conn = sqlite3.connect('profile_images.db')
        c = conn.cursor()
        c.execute("INSERT INTO images (name, image_path) VALUES (?, ?)", 
                 (name, image_path))
        conn.commit()
        last_id = c.lastrowid
        conn.close()
        return last_id
    except Exception as e:
        logger.error(f"Error storing metadata: {str(e)}")
        return None

def store_analysis(name, image_path, analysis_result):
    """Store analysis results in database"""
    try:
        conn = sqlite3.connect('profile_images.db')
        c = conn.cursor()
        
        # Store main analysis
        c.execute('''
            INSERT INTO images (name, image_path, race, age, gender, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            name,
            image_path,
            analysis_result['race'],
            analysis_result['age'],
            analysis_result['gender'],
            analysis_result['confidence']
        ))
        
        # Get the image ID
        image_id = c.lastrowid
        
        # Store race distribution
        for race, percentage in analysis_result['detailed_analysis'].get('Detailed Distribution', {}).items():
            confidence = float(percentage.rstrip('%')) / 100
            c.execute('''
                INSERT INTO race_analysis (image_id, race, confidence, is_sub_ethnicity, parent_race)
                VALUES (?, ?, ?, ?, ?)
            ''', (image_id, race, confidence, 0, None))
        
        # Store sub-ethnicities
        for race, sub_ethnicities in analysis_result['detailed_analysis'].get('Sub Ethnicities', {}).items():
            for sub_eth, sub_percentage in sub_ethnicities.items():
                confidence = float(sub_percentage.rstrip('%')) / 100
                c.execute('''
                    INSERT INTO race_analysis (image_id, race, confidence, is_sub_ethnicity, parent_race)
                    VALUES (?, ?, ?, ?, ?)
                ''', (image_id, sub_eth, confidence, 1, race))
        
        conn.commit()
        conn.close()
        logger.info(f"Successfully stored analysis for {name}")
        
    except Exception as e:
        logger.error(f"Error storing analysis: {str(e)}")
        if 'conn' in locals():
            conn.close()

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()

@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse('static/index.html')

@app.post("/clear-db/")
async def clear_database():
    """Clear the database and remove all profile images"""
    try:
        # Clear database tables
        conn = sqlite3.connect('profile_images.db')
        c = conn.cursor()
        c.execute("DELETE FROM race_analysis")
        c.execute("DELETE FROM images")
        conn.commit()
        conn.close()
        
        # Clear profile images directory
        for file in IMAGES_DIR.glob("*"):
            if file.is_file() and file.name != '.gitkeep':
                file.unlink()
        
        return JSONResponse(content={"message": "Database and images cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Create profile_images directory if it doesn't exist
        if not os.path.exists('profile_images'):
            os.makedirs('profile_images')
            
        # Read and process CSV
        content = await file.read()
        csv_data = pd.read_csv(BytesIO(content))
        results = []
        
        for index, row in csv_data.iterrows():
            name = row['name']
            profile_url = row['profile_url']
            
            # Download and save image
            try:
                image_path = f"profile_images/{name.replace(' ', '_')}_{index}.jpg"
                
                # Download image from Google Drive if it's a drive link
                if 'drive.google.com' in profile_url:
                    file_id = profile_url.split('/')[5]
                    download_url = f'https://drive.google.com/uc?id={file_id}'
                else:
                    download_url = profile_url
                    
                response = requests.get(download_url)
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Processing image: {image_path}")
                    
                    # Analyze face
                    analysis_result = analyze_face(image_path)
                    if analysis_result:
                        store_analysis(name, image_path, analysis_result)
                        results.append({
                            "name": name,
                            "success": True,
                            "analysis": analysis_result
                        })
                    else:
                        results.append({
                            "name": name,
                            "success": False,
                            "error": "Face analysis failed"
                        })
                else:
                    results.append({
                        "name": name,
                        "success": False,
                        "error": f"Failed to download image: HTTP {response.status_code}"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing {name}: {str(e)}")
                results.append({
                    "name": name,
                    "success": False,
                    "error": str(e)
                })
                
        return results
        
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/profiles/")
async def get_profiles():
    """Get all analyzed profiles"""
    try:
        conn = sqlite3.connect('profile_images.db')
        c = conn.cursor()
        
        # Get basic profile info
        c.execute("SELECT id, name, image_path, race, age, gender, confidence FROM images")
        profiles = []
        for row in c.fetchall():
            profile_id = row[0]
            
            # Get race details for this profile
            c.execute("SELECT race, confidence FROM race_analysis WHERE image_id=? ORDER BY confidence DESC", 
                     (profile_id,))
            race_details = {race: f"{conf:.1f}%" for race, conf in c.fetchall()}
            
            profiles.append({
                'id': profile_id,
                'name': row[1],
                'image_path': row[2],
                'race': row[3],
                'age': row[4],
                'gender': row[5],
                'confidence': row[6],
                'race_details': race_details
            })
        
        conn.close()
        return JSONResponse(content=profiles)
    except Exception as e:
        logger.error(f"Error fetching profiles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_images(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Analyze the image
        result = analyze_face(file_path)
        
        if result:
            # Debug logging
            print("\nDetailed Analysis Results:")
            print(f"Full result: {json.dumps(result, indent=2)}")
            
            return {
                "success": True,
                "analysis": {
                    "age": result["age"],
                    "gender": result["gender"],
                    "race": result["race"],
                    "confidence": result["confidence"],
                    "detailed_analysis": result["detailed_analysis"]
                }
            }
        else:
            return {"success": False, "error": "Analysis failed"}
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return {"success": False, "error": str(e)}
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

# Add route to serve images
@app.get("/profile_images/{image_name}")
async def get_image(image_name: str):
    image_path = f"profile_images/{image_name}"
    if os.path.exists(image_path):
        return FileResponse(image_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)