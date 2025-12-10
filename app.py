"""
10-4 Fuel Bot Agent API
A natural language fuel price query agent for iOS apps
"""

import os
import json
import pandas as pd
import math
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from anthropic import Anthropic
import uvicorn

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

# Initialize geocoder
geolocator = Nominatim(user_agent="fuel_bot_agent")

# Initialize Anthropic client
client = Anthropic()

# System prompt for the fuel bot
SYSTEM_PROMPT = """You are the 10-4 Fuel Bot, an AI assistant for fuel price information.

📊 RESPONSE FORMAT:
Always provide:
1. A markdown table with columns: Station Name | Price/Gal | Distance (mi) | City, State
2. A brief summary with top recommendation
3. For route queries, mention the route and highlight stations along the way

📍 IMPORTANT RULES:
- The CSV data has been pre-filtered and pre-sorted for you
- When price filtering is active, data is sorted by price (lowest first)
- When no price filter, data is sorted by distance (closest first)
- For route queries, stations are filtered to those along or near the route
- Show ALL stations from the CSV in your table IN THE SAME ORDER
- DO NOT re-sort the data - keep the order as provided
- Use the "DistanceFromUser" column values as-is (do NOT recalculate)
- Format prices with $ symbol (e.g., $3.99)
- Keep your response concise and helpful

Example response:
| Station Name | Price/Gal | Distance (mi) | City, State |
|--------------|-----------|---------------|-------------|
| QUIKTRIP 7134 | $3.29 | 10.5 | Atlanta, GA |
| RACETRAC #688 | $3.69 | 15.2 | Macon, GA |

**Top Recommendation:** QUIKTRIP 7134 offers the best price at $3.29/gal, just 10.5 miles away in Atlanta, GA."""

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    user_location: Optional[str] = None  # City, State format
    user_lat: Optional[float] = None
    user_lon: Optional[float] = None
    max_results: Optional[int] = 10
    max_distance_miles: Optional[float] = None
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

# Cache for geocoded locations
location_cache: Dict[str, tuple] = {}

def geocode_location(location: str) -> Optional[tuple]:
    """Convert city, state to coordinates"""
    if location in location_cache:
        return location_cache[location]
    
    try:
        loc = geolocator.geocode(location + ", USA", timeout=10)
        if loc:
            coords = (loc.latitude, loc.longitude)
            location_cache[location] = coords
            return coords
    except Exception as e:
        print(f"Geocoding error for {location}: {e}")
    return None

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
        # Read the CSV with tab delimiter based on the provided data format
        df = pd.read_csv(csv_path, sep='\t')
        
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
    """Add coordinates to stations dataframe"""
    coords = []
    for _, row in df.iterrows():
        location = f"{row['TSCity']}, {row['TSState']}"
        coord = geocode_location(location)
        if coord:
            coords.append({'lat': coord[0], 'lon': coord[1]})
        else:
            coords.append({'lat': None, 'lon': None})
    
    coords_df = pd.DataFrame(coords)
    df['Latitude'] = coords_df['lat']
    df['Longitude'] = coords_df['lon']
    return df

def filter_stations_by_distance(
    df: pd.DataFrame, 
    user_coords: tuple, 
    max_distance: float = None
) -> pd.DataFrame:
    """Filter and sort stations by distance from user"""
    distances = []
    for _, row in df.iterrows():
        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
            station_coords = (row['Latitude'], row['Longitude'])
            dist = calculate_distance(user_coords, station_coords)
            distances.append(dist)
        else:
            distances.append(float('inf'))
    
    df = df.copy()
    df['DistanceFromUser'] = distances
    
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
    extraction_prompt = f"""Extract location information from this fuel query. Return a JSON object with:
- "user_location": the user's current location (city, state) if mentioned
- "destination": destination location if this is a route query
- "is_route_query": true if asking about fuel between two locations
- "price_filter": any price constraints mentioned (e.g., "under $4")
- "fuel_type": "diesel" or "regular" if specified, otherwise "diesel"

Query: "{query}"

Return ONLY valid JSON, no explanation."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": extraction_prompt}]
        )
        
        result = json.loads(response.content[0].text)
        return result
    except Exception as e:
        print(f"Error extracting locations: {e}")
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
    """Load fuel data on startup"""
    global fuel_data
    
    # Check for fuel data file
    csv_paths = ["fuel_data.csv", "fuel_data.tsv", "/home/claude/fuel-agent/fuel_data.csv"]
    
    for path in csv_paths:
        if os.path.exists(path):
            fuel_data = load_fuel_data(path)
            if not fuel_data.empty:
                print(f"Loaded {len(fuel_data)} unique stations from {path}")
                # Pre-geocode stations (this may take a while on first run)
                fuel_data = geocode_stations(fuel_data)
                print("Geocoding complete")
                break
    
    if fuel_data is None or fuel_data.empty:
        print("Warning: No fuel data loaded. Please provide fuel_data.csv")
        fuel_data = pd.DataFrame()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "10-4 Fuel Bot API",
        "version": "1.0.0",
        "stations_loaded": len(fuel_data) if fuel_data is not None else 0
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
    """Process a natural language fuel query"""
    global fuel_data
    
    if fuel_data is None or fuel_data.empty:
        raise HTTPException(status_code=503, detail="Fuel data not loaded")
    
    # Extract location info from the query
    extracted = extract_locations_from_query(request.query)
    
    # Determine user coordinates
    user_coords = None
    if request.user_lat and request.user_lon:
        user_coords = (request.user_lat, request.user_lon)
    elif request.user_location:
        user_coords = geocode_location(request.user_location)
    elif extracted.get("user_location"):
        user_coords = geocode_location(extracted["user_location"])
    
    # Start with full dataset
    filtered_data = fuel_data.copy()
    
    # Handle route queries
    if extracted.get("is_route_query") and extracted.get("destination") and user_coords:
        dest_coords = geocode_location(extracted["destination"])
        if dest_coords:
            filtered_data = filter_stations_on_route(
                filtered_data, 
                user_coords, 
                dest_coords,
                max_deviation=request.max_distance_miles or 50
            )
            # Calculate distance from start for route ordering
            if not filtered_data.empty:
                filtered_data = filter_stations_by_distance(filtered_data, user_coords)
    
    # Filter by distance from user
    elif user_coords:
        filtered_data = filter_stations_by_distance(
            filtered_data, 
            user_coords,
            max_distance=request.max_distance_miles
        )
    
    # Sort by price if price filtering mentioned
    if extracted.get("price_filter"):
        filtered_data = filtered_data.sort_values('PumpPrice')
    
    # Limit results
    filtered_data = filtered_data.head(request.max_results or 10)
    
    # Prepare context for Claude
    data_context = prepare_data_for_llm(filtered_data, request.max_results or 10)
    
    # Generate response using Claude
    user_message = f"""User query: {request.query}

User location: {request.user_location or extracted.get('user_location', 'Not specified')}
Query type: {'Route query' if extracted.get('is_route_query') else 'Nearby search'}
Destination: {extracted.get('destination', 'N/A')}

Filtered fuel station data (CSV format):
{data_context}

Please provide a helpful response with the fuel station information in a markdown table format."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
        
        llm_response = response.content[0].text
    except Exception as e:
        llm_response = f"Error generating response: {str(e)}"
    
    # Prepare station list for response
    stations = []
    for _, row in filtered_data.iterrows():
        stations.append(StationInfo(
            station_name=row['TSName'],
            city=row['TSCity'],
            state=row['TSState'],
            price=round(row['PumpPrice'], 4),
            distance_miles=round(row['DistanceFromUser'], 1) if 'DistanceFromUser' in row and pd.notna(row['DistanceFromUser']) else None,
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
