# API Testing Guide

## Quick Test Commands

Replace `YOUR_RAILWAY_URL` with your actual Railway deployment URL.

### 1. Health Check
```bash
curl https://YOUR_RAILWAY_URL/
```

Expected response:
```json
{
  "status": "online",
  "service": "10-4 Fuel Bot API",
  "version": "1.0.0",
  "stations_loaded": 57
}
```

### 2. Query with City/State Location
```bash
curl -X POST https://YOUR_RAILWAY_URL/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find cheap diesel near me",
    "user_location": "Atlanta, GA",
    "max_results": 5
  }'
```

### 3. Query with Latitude/Longitude
```bash
curl -X POST https://YOUR_RAILWAY_URL/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me fuel prices",
    "user_lat": 33.7490,
    "user_lon": -84.3880,
    "max_results": 5
  }'
```

### 4. Query with Location in Query String
```bash
curl -X POST https://YOUR_RAILWAY_URL/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Get me fuel prices near Los Angeles, California",
    "max_results": 5
  }'
```

### 5. Route Query
```bash
curl -X POST https://YOUR_RAILWAY_URL/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me fuel stations between Atlanta and Miami",
    "user_location": "Atlanta, GA",
    "max_results": 10
  }'
```

### 6. Nearest Stations Endpoint
```bash
curl -X POST "https://YOUR_RAILWAY_URL/nearest?city=Chicago&state=IL&limit=5&max_distance_miles=100"
```

### 7. List All Stations
```bash
curl https://YOUR_RAILWAY_URL/stations?limit=10
```

## Testing with Python

```python
import requests
import json

BASE_URL = "https://YOUR_RAILWAY_URL"

# Test 1: Health Check
response = requests.get(f"{BASE_URL}/")
print("Health Check:", response.json())

# Test 2: Query with location
payload = {
    "query": "Find cheap diesel near me",
    "user_location": "Atlanta, GA",
    "max_results": 5
}
response = requests.post(f"{BASE_URL}/query", json=payload)
data = response.json()
print(f"\nFound {len(data['stations'])} stations")
print(f"Response preview: {data['response'][:200]}...")

# Test 3: Query with coordinates
payload = {
    "query": "Show me fuel prices",
    "user_lat": 33.7490,
    "user_lon": -84.3880,
    "max_results": 5
}
response = requests.post(f"{BASE_URL}/query", json=payload)
print(f"\nStatus: {response.status_code}")
```

## Expected Behavior

### ✅ Successful Query Response
- Status code: 200
- Response includes:
  - `response`: Markdown formatted table with station info
  - `stations`: Array of station objects with name, city, state, price, distance
  - `query_interpreted`: JSON string showing how the query was understood

### ❌ Error Cases

**No location provided:**
- Status code: 400
- Message: "Please provide a location using user_location (city, state) or user_lat/user_lon coordinates"

**Invalid location:**
- Status code: 400
- Message: "Could not geocode location: [location]"

**No stations found:**
- Status code: 404
- Message: "No fuel stations found near [location]"

**Service not ready:**
- Status code: 503
- Message: "Fuel data not loaded"

## Verifying Results

When testing queries, verify:

1. **Stations are from CSV data**: Check that station names match those in `fuel_data.csv`
2. **Distances are calculated**: Each station should have a `distance_miles` value
3. **Sorted by distance**: Stations should be ordered from closest to farthest (unless price filter is used)
4. **Prices are accurate**: Prices should match the `PumpPrice` column in the CSV
5. **Location is geocoded**: The response should show the interpreted location

## Common Issues

### Issue: "Could not geocode location"
**Solution**: Make sure the location format is "City, State" (e.g., "Atlanta, GA" not "Atlanta Georgia")

### Issue: "No fuel stations found"
**Solution**: Increase `max_distance_miles` parameter. The sample data only has ~57 stations across limited states.

### Issue: Slow first query
**Solution**: This is normal - the first query geocodes station locations. Subsequent queries will be faster due to caching.

### Issue: 503 Service Unavailable
**Solution**: The app is still starting up. Wait 10-20 seconds and try again.

