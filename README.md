# 🚛 10-4 Fuel Bot Agent API

A natural language fuel price query agent designed for iOS apps. Ask questions like "Find cheap gas near Atlanta" or "What are fuel prices between Dallas and Houston?" and get intelligent, formatted responses.

## Features

- **Natural Language Queries**: Ask questions in plain English
- **Location-Aware**: Geocodes city/state to coordinates automatically
- **Route Planning**: Find fuel stations along your route
- **Distance Calculations**: Sorts by proximity to your location
- **Price Filtering**: Filter by price constraints
- **LLM-Powered**: Uses Claude for intelligent query understanding

## API Endpoints

### Health Check
```
GET /
```
Returns service status and loaded station count.

### Natural Language Query
```
POST /query
```

**Request Parameters:**
- `query` (required): Natural language query string
- `user_location` (optional): City, State format (e.g., "Atlanta, GA")
- `user_lat` (optional): Latitude coordinate
- `user_lon` (optional): Longitude coordinate
- `max_results` (optional): Maximum number of results (default: 10)
- `max_distance_miles` (optional): Maximum distance to search in miles
- `fuel_type` (optional): "diesel" or "regular" (default: "diesel")

**Location Priority:**
1. If `user_lat` and `user_lon` are provided, they are used
2. If `user_location` is provided, it's geocoded to coordinates
3. If neither is provided, the query is analyzed to extract location

**Request Body Examples:**

Using City/State:
```json
{
  "query": "Find cheap diesel near me",
  "user_location": "Atlanta, GA",
  "max_results": 10,
  "max_distance_miles": 100
}
```

Using Coordinates:
```json
{
  "query": "Show me fuel prices",
  "user_lat": 33.7490,
  "user_lon": -84.3880,
  "max_results": 5
}
```

Using Query Extraction (location in query):
```json
{
  "query": "Get me fuel prices near Ogden, Utah",
  "max_results": 10
}
```

**Response:**
```json
{
  "response": "| Station Name | Price/Gal | Distance (mi) | City, State |\n|---|---|---|---|\n| RACETRAC #688 | $3.70 | 15.2 | Lithia Springs, GA |\n...",
  "stations": [
    {
      "station_name": "RACETRAC #688",
      "city": "Lithia Springs",
      "state": "GA",
      "price": 3.6987,
      "distance_miles": 15.2,
      "latitude": 33.7940,
      "longitude": -84.6630
    }
  ],
  "query_interpreted": "{\"user_location\": \"Atlanta, GA\", \"is_route_query\": false, ...}"
}
```

### Find Nearest Stations
```
POST /nearest?city=Atlanta&state=GA&limit=5&max_distance_miles=100
```

### Find Stations on Route
```
POST /route?start_city=Atlanta&start_state=GA&end_city=Jacksonville&end_state=FL&limit=10
```

### List All Stations
```
GET /stations?limit=20
```

## iOS Integration Example

```swift
import Foundation

struct FuelQuery: Codable {
    let query: String
    let userLocation: String?
    let maxResults: Int?
    let maxDistanceMiles: Double?
    
    enum CodingKeys: String, CodingKey {
        case query
        case userLocation = "user_location"
        case maxResults = "max_results"
        case maxDistanceMiles = "max_distance_miles"
    }
}

struct FuelResponse: Codable {
    let response: String
    let stations: [Station]
    let queryInterpreted: String
    
    enum CodingKeys: String, CodingKey {
        case response
        case stations
        case queryInterpreted = "query_interpreted"
    }
}

struct Station: Codable {
    let stationName: String
    let city: String
    let state: String
    let price: Double
    let distanceMiles: Double?
    
    enum CodingKeys: String, CodingKey {
        case stationName = "station_name"
        case city
        case state
        case price
        case distanceMiles = "distance_miles"
    }
}

class FuelBotClient {
    let baseURL: String
    
    init(baseURL: String) {
        self.baseURL = baseURL
    }
    
    func queryFuel(query: String, location: String?, completion: @escaping (Result<FuelResponse, Error>) -> Void) {
        guard let url = URL(string: "\(baseURL)/query") else { return }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body = FuelQuery(
            query: query,
            userLocation: location,
            maxResults: 10,
            maxDistanceMiles: 100
        )
        
        request.httpBody = try? JSONEncoder().encode(body)
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let data = data else { return }
            
            do {
                let response = try JSONDecoder().decode(FuelResponse.self, from: data)
                completion(.success(response))
            } catch {
                completion(.failure(error))
            }
        }.resume()
    }
}

// Usage
let client = FuelBotClient(baseURL: "https://your-app.railway.app")
client.queryFuel(query: "Find cheap gas near me", location: "Atlanta, GA") { result in
    switch result {
    case .success(let response):
        print(response.response)
    case .failure(let error):
        print("Error: \(error)")
    }
}
```

## Deployment

### Option 1: Railway (Recommended - Free Tier)

1. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway:**
   ```bash
   railway login
   ```

3. **Initialize and deploy:**
   ```bash
   cd fuel-agent
   railway init
   railway up
   ```

4. **Set environment variable:**
   ```bash
   railway variables set ANTHROPIC_API_KEY=your-api-key
   ```

5. **Get your public URL:**
   ```bash
   railway domain
   ```

### Option 2: Render.com

1. Push code to GitHub
2. Go to [render.com](https://render.com) and create new Web Service
3. Connect your GitHub repo
4. Set environment variables:
   - `ANTHROPIC_API_KEY`: Your Anthropic API key
5. Deploy!

### Option 3: Docker (Local/Self-hosted)

```bash
# Build
docker build -t fuel-bot .

# Run
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your-key fuel-bot
```

### Option 4: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=your-api-key

# Run
python app.py
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key for Claude |
| `PORT` | No | Port to run on (default: 8000) |

## Example Queries

### Simple Location Queries
- "Find the cheapest diesel near me" (with `user_location: "Atlanta, GA"`)
- "Show me fuel prices" (with `user_lat: 33.7490, user_lon: -84.3880`)
- "Get me fuel prices near Ogden, Utah" (location extracted from query)

### Route Queries
- "What are fuel prices between Dallas and Houston?"
- "Find fuel stops along my route from Los Angeles to Phoenix"

### Price Filtering
- "Show me gas stations under $4 per gallon near Chicago"
- "Find the cheapest diesel in California"

### Specific Stations
- "What's the nearest Pilot station to Indianapolis?"
- "Show me Love's Travel Stops near me"

## Testing the API

A test script is included to verify all endpoints work correctly:

```bash
# Make sure the API is running locally
python app.py

# In another terminal, run the tests
pip install requests
python test_api.py
```

The test script will verify:
- Health check endpoint
- Query with city/state location
- Query with lat/lon coordinates
- Query for locations without nearby stations
- Nearest stations endpoint
- Error handling for missing location

## Data Format

The fuel data CSV should have these columns:
- `TSName` - Station name
- `TSCity` - City
- `TSState` - State (2-letter code)
- `PumpPrice` - Price per gallon
- `PROD` - Product type (ULSD=Diesel, ULSR=Regular)

## License

MIT License
