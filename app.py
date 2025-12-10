"""
10-4 Fuel Bot Agent API
A natural language fuel price query agent for iOS apps
"""

import os
import json
import pandas as pd
import math
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from geopy.distance import geodesic
from anthropic import Anthropic
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="10-4 Fuel Bot API",
    description="Natural language fuel price query agent",
    version="1.0.0"
)

# Enable CORS for iOS app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Geocoding API keys (optional - uses multiple providers for reliability)
GEOCODIO_API_KEY = os.environ.get("GEOCODIO_API_KEY", "")
OPENCAGE_API_KEY = os.environ.get("OPENCAGE_API_KEY", "")

# Initialize Anthropic client
client = Anthropic()

# System prompt for the fuel bot
SYSTEM_PROMPT = """You are the 10-4 Fuel Bot. Provide fuel station info in this format:

| Station Name | Price/Gal | Distance (mi) | City, State |
|--------------|-----------|---------------|-------------|
| STATION | $X.XX | Y.Y | City, ST |

Then add a brief recommendation (1-2 sentences max).

RULES:
- Show ALL stations from CSV in SAME ORDER
- DO NOT repeat info from table in text
- Format prices with $ (e.g., $3.99)
- Keep response concise"""

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    user_location: Optional[str] = None  # City, State format
    user_lat: Optional[float] = None
    user_lon: Optional[float] = None
    max_results: Optional[int] = 10
    max_distance_miles: Optional[float] = 50.0  # Default 50 miles radius
    fuel_type: Optional[str] = "diesel"  # diesel or regular

class StationInfo(BaseModel):
    station_name: str
    city: str
    state: str
    price: float
    distance_miles: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class QueryResponse(BaseModel):
    response: str
    stations: List[StationInfo]
    query_interpreted: str

# Cache for geocoded locations (persistent)
GEOCODE_CACHE_FILE = "geocode_cache.json"
location_cache: Dict[str, tuple] = {}

def load_geocode_cache():
    """Load geocode cache from file"""
    global location_cache

    # Debug: print current working directory and file existence
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    print(f"Looking for geocode cache at: {GEOCODE_CACHE_FILE}")
    print(f"File exists: {os.path.exists(GEOCODE_CACHE_FILE)}")

    # Try multiple possible locations
    possible_paths = [
        GEOCODE_CACHE_FILE,
        os.path.join(cwd, GEOCODE_CACHE_FILE),
        "/app/geocode_cache.json",  # Railway default
        "./geocode_cache.json"
    ]

    cache_loaded = False
    for path in possible_paths:
        if os.path.exists(path):
            try:
                print(f"Found cache file at: {path}")
                with open(path, 'r') as f:
                    data = json.load(f)
                    # Convert lists back to tuples
                    location_cache = {k: tuple(v) for k, v in data.items()}
                    print(f"✅ Loaded {len(location_cache)} cached locations from {path}")
                    cache_loaded = True
                    break
            except Exception as e:
                print(f"Error loading geocode cache from {path}: {e}")

    if not cache_loaded:
        print(f"⚠️  No geocode cache file found. Checked paths: {possible_paths}")
        location_cache = {}

def save_geocode_cache():
    """Save geocode cache to file"""
    try:
        # Convert tuples to lists for JSON serialization
        data = {k: list(v) for k, v in location_cache.items()}
        with open(GEOCODE_CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(location_cache)} locations to {GEOCODE_CACHE_FILE}")
    except Exception as e:
        print(f"Error saving geocode cache: {e}")

def geocode_location(location: str) -> Optional[tuple]:
    """Convert city, state to coordinates using free geocoding APIs with persistent caching"""
    # Check cache first (from file)
    if location in location_cache:
        print(f"📍 Cache hit: {location} -> {location_cache[location]}")
        return location_cache[location]

    coords = None

    # Try Geocodio first (2,500 free requests/day, very fast)
    if GEOCODIO_API_KEY:
        try:
            url = "https://api.geocod.io/v1.7/geocode"
            params = {
                "q": f"{location}, USA",
                "api_key": GEOCODIO_API_KEY
            }
            response = requests.get(url, params=params, timeout=5)
            data = response.json()

            if data.get("results") and len(data["results"]) > 0:
                result = data["results"][0]
                lat = result["location"]["lat"]
                lng = result["location"]["lng"]
                coords = (lat, lng)
                print(f"✓ Geocodio: {location} -> {coords}")
        except Exception as e:
            print(f"Geocodio error for {location}: {e}")

    # Try OpenCage as fallback (2,500 free requests/day)
    if not coords and OPENCAGE_API_KEY:
        try:
            url = "https://api.opencagedata.com/geocode/v1/json"
            params = {
                "q": f"{location}, USA",
                "key": OPENCAGE_API_KEY,
                "limit": 1
            }
            response = requests.get(url, params=params, timeout=5)
            data = response.json()

            if data.get("results") and len(data["results"]) > 0:
                result = data["results"][0]
                lat = result["geometry"]["lat"]
                lng = result["geometry"]["lng"]
                coords = (lat, lng)
                print(f"✓ OpenCage: {location} -> {coords}")
        except Exception as e:
            print(f"OpenCage error for {location}: {e}")

    # If no API keys are set, return None
    if not coords and not GEOCODIO_API_KEY and not OPENCAGE_API_KEY:
        print(f"⚠️  No geocoding API keys configured. Please add GEOCODIO_API_KEY or OPENCAGE_API_KEY to .env")

    # Save to cache if we got coordinates
    if coords:
        location_cache[location] = coords
        save_geocode_cache()  # Persist to file immediately

    return coords

def calculate_distance(coord1: tuple, coord2: tuple) -> float:
    """Calculate distance in miles between two coordinates"""
    return geodesic(coord1, coord2).miles

def is_point_near_route(point: tuple, start: tuple, end: tuple, max_deviation_miles: float = 50) -> bool:
    """Check if a point is near the route between start and end"""
    # Simple approach: check if point is within a corridor between start and end
    # More sophisticated would use actual route APIs
    
    # Calculate distances
    d_start = calculate_distance(start, point)
    d_end = calculate_distance(end, point)
    d_route = calculate_distance(start, end)
    
    # Point is "on route" if sum of distances to endpoints is close to route distance
    # Allow some deviation for realistic routing
    return (d_start + d_end) <= (d_route + max_deviation_miles * 2)

def load_fuel_data(csv_path: str = "fuel_data.csv") -> pd.DataFrame:
    """Load and preprocess fuel data from CSV"""
    try:
        # Read the CSV (comma-delimited format)
        df = pd.read_csv(csv_path)
        
        # Clean up column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Filter to only records with valid fuel prices
        df = df[df['PumpPrice'] > 0].copy()
        
        # Clean city and state names
        df['TSCity'] = df['TSCity'].str.strip()
        df['TSState'] = df['TSState'].str.strip()
        
        # Create location string for geocoding
        df['Location'] = df['TSCity'] + ', ' + df['TSState']
        
        # Get unique stations with their latest prices
        stations = df.groupby(['TSName', 'TSCity', 'TSState']).agg({
            'PumpPrice': 'mean',  # Average price for the station
            'PROD': 'first',
            'Location': 'first'
        }).reset_index()
        
        return stations
    except Exception as e:
        print(f"Error loading fuel data: {e}")
        return pd.DataFrame()

def geocode_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Add coordinates to stations dataframe (lazy geocoding)"""
    # Only geocode stations that don't have coordinates yet
    for idx, row in df.iterrows():
        if pd.isna(row.get('Latitude')) or pd.isna(row.get('Longitude')):
            location = f"{row['TSCity']}, {row['TSState']}"
            coord = geocode_location(location)
            if coord:
                df.at[idx, 'Latitude'] = coord[0]
                df.at[idx, 'Longitude'] = coord[1]
    return df

def filter_stations_by_distance(
    df: pd.DataFrame,
    user_coords: tuple,
    max_distance: float = None
) -> pd.DataFrame:
    """Filter and sort stations by distance from user"""
    # Geocode stations that need it
    df = geocode_stations(df)

    distances = []
    for _, row in df.iterrows():
        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
            station_coords = (row['Latitude'], row['Longitude'])
            dist = calculate_distance(user_coords, station_coords)
            distances.append(dist)
        else:
            # Use None instead of inf for stations without coordinates
            distances.append(None)

    df = df.copy()
    df['DistanceFromUser'] = distances

    # Remove stations without valid coordinates (distance is None)
    df = df[df['DistanceFromUser'].notna()]

    # Filter by max distance if specified
    if max_distance:
        df = df[df['DistanceFromUser'] <= max_distance]

    # Sort by distance
    df = df.sort_values('DistanceFromUser')

    return df

def filter_stations_on_route(
    df: pd.DataFrame,
    start_coords: tuple,
    end_coords: tuple,
    max_deviation: float = 50
) -> pd.DataFrame:
    """Filter stations that are along a route"""
    on_route = []
    for _, row in df.iterrows():
        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
            station_coords = (row['Latitude'], row['Longitude'])
            if is_point_near_route(station_coords, start_coords, end_coords, max_deviation):
                on_route.append(True)
            else:
                on_route.append(False)
        else:
            on_route.append(False)
    
    df = df.copy()
    df['OnRoute'] = on_route
    return df[df['OnRoute'] == True]

def extract_locations_from_query(query: str) -> Dict[str, Any]:
    """Use Claude to extract location information from the query"""
    extraction_prompt = f"""Extract location information from this fuel query and return ONLY a JSON object.

Query: "{query}"

Return JSON with these fields:
- "user_location": user's current location (city, state format) or null. Use this for "near X", "in X", "at X" queries.
- "destination": destination location (city, state format) or null. Only use for route queries with "between X and Y".
- "is_route_query": true if asking about fuel between two locations, false otherwise
- "price_filter": any price constraints mentioned or null
- "fuel_type": "diesel" or "regular" if specified, otherwise "diesel"

Examples:
- "Get fuel prices near Ogden, UT" -> {{"user_location": "Ogden, UT", "destination": null, "is_route_query": false, "price_filter": null, "fuel_type": "diesel"}}
- "Get fuel between Atlanta and Florida" -> {{"user_location": "Atlanta, GA", "destination": "Florida", "is_route_query": true, "price_filter": null, "fuel_type": "diesel"}}

Return ONLY the JSON object, nothing else."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system="You are a JSON extraction tool. Return ONLY valid JSON, no explanation or markdown.",
            messages=[{"role": "user", "content": extraction_prompt}]
        )

        response_text = response.content[0].text.strip()
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        result = json.loads(response_text)
        print(f"✓ Extracted: {result}")
        return result
    except Exception as e:
        print(f"Error extracting locations: {e}")
        print(f"Response was: {response.content[0].text if 'response' in locals() else 'No response'}")
        return {
            "user_location": None,
            "destination": None,
            "is_route_query": False,
            "price_filter": None,
            "fuel_type": "diesel"
        }

def prepare_data_for_llm(df: pd.DataFrame, max_results: int = 10) -> str:
    """Convert dataframe to CSV string for LLM context"""
    df_limited = df.head(max_results).copy()
    
    # Format for readability
    output_df = pd.DataFrame({
        'StationName': df_limited['TSName'],
        'City': df_limited['TSCity'],
        'State': df_limited['TSState'],
        'PricePerGallon': df_limited['PumpPrice'].round(4),
        'DistanceFromUser': df_limited['DistanceFromUser'].round(1) if 'DistanceFromUser' in df_limited.columns else 'N/A'
    })
    
    return output_df.to_csv(index=False)

# Global fuel data (loaded on startup)
fuel_data: pd.DataFrame = None

@app.on_event("startup")
async def startup_event():
    """Load fuel data and geocode cache on startup"""
    global fuel_data

    # Load geocode cache from file
    load_geocode_cache()

    # Check for fuel data file
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")

    csv_paths = [
        "fuel_data.csv",
        "fuel_data.tsv",
        "/home/claude/fuel-agent/fuel_data.csv",
        "/app/fuel_data.csv",  # Railway default
        os.path.join(cwd, "fuel_data.csv")
    ]

    print(f"Looking for CSV in these paths: {csv_paths}")

    for path in csv_paths:
        print(f"Checking path: {path} - exists: {os.path.exists(path)}")
        if os.path.exists(path):
            fuel_data = load_fuel_data(path)
            if not fuel_data.empty:
                print(f"Loaded {len(fuel_data)} unique stations from {path}")

                # Pre-geocode all stations at startup for better performance
                print("Pre-geocoding all stations...")
                fuel_data = geocode_stations(fuel_data)

                # Count how many stations have coordinates
                geocoded_count = fuel_data[['Latitude', 'Longitude']].notna().all(axis=1).sum()
                print(f"✅ Pre-geocoded {geocoded_count}/{len(fuel_data)} stations successfully")
                print("Fuel data loaded and ready!")
                break

    if fuel_data is None or fuel_data.empty:
        print("⚠️  WARNING: No fuel data loaded. Please provide fuel_data.csv")
        print(f"Files in current directory: {os.listdir(cwd)}")
        fuel_data = pd.DataFrame()

@app.get("/")
async def root():
    """Health check endpoint"""
    print("ROOT ENDPOINT CALLED!")  # Debug log
    return {
        "status": "online",
        "service": "10-4 Fuel Bot API",
        "version": "1.0.0",
        "stations_loaded": len(fuel_data) if fuel_data is not None else 0
    }

@app.get("/test")
async def test():
    """Simple test endpoint"""
    print("TEST ENDPOINT CALLED!")  # Debug log
    return {"status": "ok", "message": "Test endpoint working"}

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check file system"""
    cwd = os.getcwd()
    files_in_cwd = os.listdir(cwd)

    csv_paths = [
        "fuel_data.csv",
        "/app/fuel_data.csv",
        os.path.join(cwd, "fuel_data.csv")
    ]

    path_checks = {path: os.path.exists(path) for path in csv_paths}

    cache_paths = [
        "geocode_cache.json",
        "/app/geocode_cache.json",
        os.path.join(cwd, "geocode_cache.json")
    ]

    cache_checks = {path: os.path.exists(path) for path in cache_paths}

    # Try to load CSV and see what happens
    csv_load_error = None
    csv_load_success = False
    try:
        test_df = pd.read_csv("fuel_data.csv")
        csv_load_success = True
        csv_rows = len(test_df)
        csv_columns = list(test_df.columns)
    except Exception as e:
        csv_load_error = str(e)
        csv_rows = 0
        csv_columns = []

    return {
        "cwd": cwd,
        "files_in_cwd": files_in_cwd,
        "csv_path_checks": path_checks,
        "cache_path_checks": cache_checks,
        "fuel_data_loaded": fuel_data is not None and not fuel_data.empty,
        "stations_count": len(fuel_data) if fuel_data is not None else 0,
        "location_cache_size": len(location_cache),
        "csv_test_load": {
            "success": csv_load_success,
            "error": csv_load_error,
            "rows": csv_rows,
            "columns": csv_columns[:10] if csv_columns else []
        }
    }

@app.get("/stations")
async def get_stations(limit: int = 20):
    """Get list of all stations"""
    if fuel_data is None or fuel_data.empty:
        raise HTTPException(status_code=503, detail="Fuel data not loaded")
    
    stations = []
    for _, row in fuel_data.head(limit).iterrows():
        stations.append(StationInfo(
            station_name=row['TSName'],
            city=row['TSCity'],
            state=row['TSState'],
            price=round(row['PumpPrice'], 4),
            latitude=row.get('Latitude'),
            longitude=row.get('Longitude')
        ))
    
    return {"stations": stations, "total": len(fuel_data)}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a natural language fuel query with parallel processing"""
    global fuel_data

    if fuel_data is None or fuel_data.empty:
        raise HTTPException(status_code=503, detail="Fuel data not loaded")

    # Debug logging
    print(f"\n{'='*60}")
    print(f"NEW QUERY RECEIVED: {request.query}")
    print(f"User location param: {request.user_location}")
    print(f"User coords: ({request.user_lat}, {request.user_lon})")
    print(f"{'='*60}\n")

    # Run location extraction in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        extracted = await loop.run_in_executor(executor, extract_locations_from_query, request.query)

    print(f"Extracted location info: {extracted}")

    # Determine user coordinates - smart logic to decide between query location vs GPS
    user_coords = None
    user_location_str = None
    location_required = True

    # Check if this is a query that doesn't need location (e.g., "lowest price")
    query_lower = request.query.lower()
    if any(keyword in query_lower for keyword in ["lowest", "cheapest", "best price", "all stations"]):
        location_required = False

    # SMART LOGIC: Decide whether to use extracted location or GPS coordinates
    # If Claude extracted a SPECIFIC location (city, state), use that - ignore GPS
    # If query is generic ("near me", "lowest price"), use GPS coordinates

    has_specific_location = extracted.get("user_location") and extracted["user_location"] is not None
    has_gps_coords = request.user_lat is not None and request.user_lon is not None

    print(f"Smart location decision:")
    print(f"  - Extracted location: {extracted.get('user_location')}")
    print(f"  - Has GPS coords: {has_gps_coords}")
    print(f"  - Query: {request.query}")

    # Priority 1: If query mentions a SPECIFIC location, use that (ignore GPS)
    if has_specific_location:
        print(f"✓ Using SPECIFIC location from query: {extracted['user_location']}")
        # Run geocoding in parallel
        with ThreadPoolExecutor() as executor:
            user_coords = await loop.run_in_executor(executor, geocode_location, extracted["user_location"])
        user_location_str = extracted["user_location"]
        print(f"  Geocoded to: {user_coords}")
        if not user_coords:
            raise HTTPException(status_code=400, detail=f"Could not geocode location: {extracted['user_location']}")

    # Priority 2: If no specific location but GPS available, use GPS
    elif has_gps_coords:
        print(f"✓ Using GPS coordinates: ({request.user_lat}, {request.user_lon})")
        user_coords = (request.user_lat, request.user_lon)
        user_location_str = f"your location ({request.user_lat:.4f}, {request.user_lon:.4f})"

    # Priority 3: Explicit user_location parameter (fallback for manual override)
    elif request.user_location:
        print(f"✓ Using explicit user_location parameter: {request.user_location}")
        # Run geocoding in parallel
        with ThreadPoolExecutor() as executor:
            user_coords = await loop.run_in_executor(executor, geocode_location, request.user_location)
        user_location_str = request.user_location
        print(f"  Geocoded to: {user_coords}")
        if not user_coords:
            raise HTTPException(status_code=400, detail=f"Could not geocode location: {request.user_location}")

    # If no location provided and it's required, return error
    if not user_coords and location_required:
        raise HTTPException(
            status_code=400,
            detail="Please provide a location using user_location (city, state) or user_lat/user_lon coordinates"
        )

    # Start with full dataset
    filtered_data = fuel_data.copy()

    # Handle route queries
    if extracted.get("is_route_query") and extracted.get("destination"):
        dest_coords = geocode_location(extracted["destination"])
        if dest_coords and user_coords:
            # For route queries, use a larger deviation (100 miles) to find stations along the way
            route_deviation = 100  # miles
            filtered_data = filter_stations_on_route(
                filtered_data,
                user_coords,
                dest_coords,
                max_deviation=route_deviation
            )
            # Calculate distance from start for route ordering
            if not filtered_data.empty:
                filtered_data = filter_stations_by_distance(filtered_data, user_coords)
        else:
            raise HTTPException(status_code=400, detail=f"Could not geocode destination: {extracted['destination']}")

    # Filter by distance from user (if location provided)
    elif user_coords:
        print(f"Filtering stations by distance from {user_coords} within {request.max_distance_miles} miles")
        filtered_data = filter_stations_by_distance(
            filtered_data,
            user_coords,
            max_distance=request.max_distance_miles
        )
        print(f"Found {len(filtered_data)} stations after distance filtering")

    # No location - just sort by price for queries like "lowest price"
    else:
        filtered_data = filtered_data.sort_values('PumpPrice')

    # Apply price filtering if specified
    if extracted.get("price_filter"):
        price_filter_str = extracted["price_filter"].lower()
        print(f"Applying price filter: {price_filter_str}")

        # Extract price threshold from filter string
        import re
        price_match = re.search(r'\$?(\d+\.?\d*)', price_filter_str)

        if price_match:
            price_threshold = float(price_match.group(1))
            print(f"Price threshold extracted: ${price_threshold}")

            # Apply the appropriate filter
            if any(keyword in price_filter_str for keyword in ["under", "less than", "below", "cheaper than"]):
                print(f"Filtering for prices UNDER ${price_threshold}")
                filtered_data = filtered_data[filtered_data['PumpPrice'] < price_threshold]
            elif any(keyword in price_filter_str for keyword in ["over", "more than", "above", "greater than"]):
                print(f"Filtering for prices OVER ${price_threshold}")
                filtered_data = filtered_data[filtered_data['PumpPrice'] > price_threshold]
            elif any(keyword in price_filter_str for keyword in ["exactly", "equal to"]):
                print(f"Filtering for prices EQUAL to ${price_threshold}")
                filtered_data = filtered_data[filtered_data['PumpPrice'] == price_threshold]

            print(f"Stations after price filtering: {len(filtered_data)}")

    # Check if we have any results
    if filtered_data.empty:
        if extracted.get("is_route_query") and extracted.get("destination"):
            raise HTTPException(
                status_code=404,
                detail=f"No fuel stations found on route from {user_location_str} to {extracted['destination']}"
            )
        elif user_location_str:
            raise HTTPException(
                status_code=404,
                detail=f"No fuel stations found near {user_location_str}" +
                       (f" within {request.max_distance_miles} miles" if request.max_distance_miles else "")
            )
        else:
            raise HTTPException(status_code=404, detail="No fuel stations found")

    # Sort by price if price filtering mentioned or no location
    if extracted.get("price_filter") or not user_coords:
        filtered_data = filtered_data.sort_values('PumpPrice')

    # Limit results
    filtered_data = filtered_data.head(request.max_results or 10)

    # Prepare context for Claude
    data_context = prepare_data_for_llm(filtered_data, request.max_results or 10)

    # Generate response using Claude
    if user_location_str:
        user_message = f"""User query: {request.query}

User location: {user_location_str}
Query type: {'Route query' if extracted.get('is_route_query') else 'Nearby search'}
Destination: {extracted.get('destination', 'N/A')}

Filtered fuel station data (CSV format):
{data_context}

Please provide a helpful response with the fuel station information in a markdown table format. Do not repeat information already in the table."""
    else:
        user_message = f"""User query: {request.query}

Query type: Showing stations sorted by price

Fuel station data (CSV format):
{data_context}

Please provide a helpful response with the fuel station information in a markdown table format. Do not repeat information already in the table."""

    # Run Claude API call in parallel to avoid blocking
    try:
        def call_claude():
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,  # Reduced for faster responses
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}]
            )
            return response.content[0].text

        with ThreadPoolExecutor() as executor:
            llm_response = await loop.run_in_executor(executor, call_claude)
    except Exception as e:
        llm_response = f"Error generating response: {str(e)}"
    
    # Prepare station list for response
    stations = []
    for _, row in filtered_data.iterrows():
        # Safely handle distance - ensure it's a valid number
        distance = None
        if 'DistanceFromUser' in row and pd.notna(row['DistanceFromUser']):
            try:
                dist_val = float(row['DistanceFromUser'])
                if math.isfinite(dist_val):  # Check if it's not inf or nan
                    distance = round(dist_val, 1)
            except (ValueError, TypeError):
                pass

        stations.append(StationInfo(
            station_name=row['TSName'],
            city=row['TSCity'],
            state=row['TSState'],
            price=round(row['PumpPrice'], 4),
            distance_miles=distance,
            latitude=row.get('Latitude') if pd.notna(row.get('Latitude')) else None,
            longitude=row.get('Longitude') if pd.notna(row.get('Longitude')) else None
        ))
    
    return QueryResponse(
        response=llm_response,
        stations=stations,
        query_interpreted=json.dumps(extracted)
    )

@app.post("/nearest")
async def find_nearest(
    city: str,
    state: str,
    limit: int = 5,
    max_distance_miles: float = 100
):
    """Find nearest fuel stations to a location"""
    if fuel_data is None or fuel_data.empty:
        raise HTTPException(status_code=503, detail="Fuel data not loaded")
    
    location = f"{city}, {state}"
    user_coords = geocode_location(location)
    
    if not user_coords:
        raise HTTPException(status_code=400, detail=f"Could not geocode location: {location}")
    
    filtered = filter_stations_by_distance(
        fuel_data, 
        user_coords, 
        max_distance=max_distance_miles
    ).head(limit)
    
    stations = []
    for _, row in filtered.iterrows():
        stations.append({
            "station_name": row['TSName'],
            "city": row['TSCity'],
            "state": row['TSState'],
            "price": round(row['PumpPrice'], 4),
            "distance_miles": round(row['DistanceFromUser'], 1)
        })
    
    return {
        "location": location,
        "coordinates": user_coords,
        "stations": stations
    }

@app.post("/route")
async def find_on_route(
    start_city: str,
    start_state: str,
    end_city: str,
    end_state: str,
    limit: int = 10,
    max_deviation_miles: float = 50
):
    """Find fuel stations along a route"""
    if fuel_data is None or fuel_data.empty:
        raise HTTPException(status_code=503, detail="Fuel data not loaded")
    
    start_location = f"{start_city}, {start_state}"
    end_location = f"{end_city}, {end_state}"
    
    start_coords = geocode_location(start_location)
    end_coords = geocode_location(end_location)
    
    if not start_coords:
        raise HTTPException(status_code=400, detail=f"Could not geocode start: {start_location}")
    if not end_coords:
        raise HTTPException(status_code=400, detail=f"Could not geocode end: {end_location}")
    
    # Filter stations on route
    on_route = filter_stations_on_route(
        fuel_data, 
        start_coords, 
        end_coords,
        max_deviation=max_deviation_miles
    )
    
    # Sort by distance from start
    on_route = filter_stations_by_distance(on_route, start_coords).head(limit)
    
    stations = []
    for _, row in on_route.iterrows():
        stations.append({
            "station_name": row['TSName'],
            "city": row['TSCity'],
            "state": row['TSState'],
            "price": round(row['PumpPrice'], 4),
            "distance_from_start_miles": round(row['DistanceFromUser'], 1)
        })
    
    return {
        "route": {
            "start": start_location,
            "end": end_location,
            "total_distance_miles": round(calculate_distance(start_coords, end_coords), 1)
        },
        "stations": stations
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug", access_log=True)
