import requests

def get_climate_data(lat: float, lon: float):
    """
    Fetches average annual/recent climate data from Open-Meteo.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # We fetch a generic recent year to get basic climate context
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "daily": ["temperature_2m_mean", "precipitation_sum"],
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        daily = data.get("daily", {})
        temps = [t for t in daily.get("temperature_2m_mean", []) if t is not None]
        precip = [p for p in daily.get("precipitation_sum", []) if p is not None]
        
        result = {}
        if temps:
            result["temperature_annual_mean"] = round(sum(temps) / len(temps), 2)
        if precip:
            result["precipitation_annual_sum"] = round(sum(precip), 2)
            
        return result if result else None
    except Exception as e:
        print(f"Error fetching Climate data: {e}")
        return None

