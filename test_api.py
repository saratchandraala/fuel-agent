"""
Test script for the Fuel Bot API
"""
import requests
import json

# Base URL - change this to your deployed URL or use localhost
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_query_with_city_state():
    """Test query with city, state"""
    print("\n=== Testing Query with City, State ===")
    payload = {
        "query": "Get me fuel prices near me",
        "user_location": "Atlanta, GA",
        "max_results": 5
    }
    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Found {len(data['stations'])} stations")
        print(f"\nFirst station: {data['stations'][0] if data['stations'] else 'None'}")
        print(f"\nLLM Response Preview: {data['response'][:200]}...")
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200

def test_query_with_lat_lon():
    """Test query with latitude/longitude"""
    print("\n=== Testing Query with Lat/Lon ===")
    # Coordinates for Chicago, IL
    payload = {
        "query": "Show me diesel prices",
        "user_lat": 41.8781,
        "user_lon": -87.6298,
        "max_results": 5
    }
    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Found {len(data['stations'])} stations")
        print(f"\nFirst station: {data['stations'][0] if data['stations'] else 'None'}")
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200

def test_query_ogden_utah():
    """Test query for Ogden, Utah (should return nearby stations even if none in Ogden)"""
    print("\n=== Testing Query for Ogden, Utah ===")
    payload = {
        "query": "Get me fuel prices near Ogden, Utah",
        "user_location": "Ogden, UT",
        "max_results": 10,
        "max_distance_miles": 500  # Wider search since no UT stations in sample data
    }
    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Found {len(data['stations'])} stations")
        if data['stations']:
            print(f"\nClosest station: {data['stations'][0]}")
            print(f"\nLLM Response Preview: {data['response'][:300]}...")
    else:
        print(f"Error: {response.text}")
    return response.status_code in [200, 404]  # 404 is acceptable if no stations nearby

def test_nearest_endpoint():
    """Test the /nearest endpoint"""
    print("\n=== Testing /nearest Endpoint ===")
    params = {
        "city": "Los Angeles",
        "state": "CA",
        "limit": 5,
        "max_distance_miles": 100
    }
    response = requests.post(f"{BASE_URL}/nearest", params=params)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Location: {data['location']}")
        print(f"Coordinates: {data['coordinates']}")
        print(f"Found {len(data['stations'])} stations")
        if data['stations']:
            print(f"\nClosest station: {data['stations'][0]}")
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200

def test_no_location_error():
    """Test that query without location returns proper error"""
    print("\n=== Testing Query Without Location (Should Error) ===")
    payload = {
        "query": "Show me fuel prices"
    }
    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    return response.status_code == 400

if __name__ == "__main__":
    print("=" * 60)
    print("Fuel Bot API Test Suite")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Query with City/State", test_query_with_city_state),
        ("Query with Lat/Lon", test_query_with_lat_lon),
        ("Query Ogden, Utah", test_query_ogden_utah),
        ("Nearest Endpoint", test_nearest_endpoint),
        ("No Location Error", test_no_location_error),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"Exception: {e}")
            results.append((test_name, "ERROR"))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for test_name, result in results:
        status_symbol = "✓" if result == "PASS" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
    
    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

