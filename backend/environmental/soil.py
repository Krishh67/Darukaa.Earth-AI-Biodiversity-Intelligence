import requests

def get_soil_data(lat: float, lon: float):
    """
    Fetches soil data from SoilGrids REST API based on coordinates.
    """
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lon": lon,
        "lat": lat,
        "property": ["soc", "phh2o", "nitrogen", "bdod", "sand", "silt", "clay"],
        "depth": ["0-5cm"],
        "value": ["mean"]
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        layers = data.get("properties", {}).get("layers", [])
        result = {}
        for layer in layers:
            name = layer.get("name")
            depths = layer.get("depths", [])
            if depths:
                val = depths[0].get("values", {}).get("mean")
                if val is not None:
                    # SoilGrids returns integers that need conversion (e.g., pH is x10, soc is dg/kg)
                    if name == "phh2o":
                        result["ph"] = val / 10.0
                    elif name == "soc":
                        result["soc"] = val / 10.0  # g/kg
                    elif name == "nitrogen":
                        result["nitrogen"] = val / 100.0 # cg/kg to g/kg
                    elif name == "bdod":
                        result["bulk_density"] = val / 100.0 # cg/cm3 to g/cm3
                    else:
                        result[name] = val / 10.0 # sand, silt, clay in % usually x10
        return result
    except Exception as e:
        print(f"Error fetching SoilGrids data: {e}")
        return None

