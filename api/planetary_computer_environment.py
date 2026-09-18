from pystac_client import Client
import planetary_computer
import rasterio

def get_landcover_data(latitude: float, longitude: float):
    try:
        CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
        catalog = Client.open(
            CATALOG_URL,
            modifier=planetary_computer.sign_inplace
        )

        search = catalog.search(
            collections=["esa-worldcover"],
            intersects={
                "type": "Point",
                "coordinates": [longitude, latitude]
            }
        )

        items = list(search.items())
        if not items:
            print("No ESA WorldCover data found for this location.")
            return None

        item = items[0]
        asset = item.assets.get("map")

        if asset is None:
            print("Could not find the WorldCover 'map' asset.")
            return None

        with rasterio.open(asset.href) as src:
            row, col = src.index(longitude, latitude)
            land_cover_value = src.read(1)[row, col]

        LAND_COVER_CLASSES = {
            10: "Tree cover",
            20: "Shrubland",
            30: "Grassland",
            40: "Cropland",
            50: "Built-up",
            60: "Bare / sparse vegetation",
            70: "Snow and ice",
            80: "Permanent water bodies",
            90: "Herbaceous wetland",
            95: "Mangroves",
            100: "Moss and lichen"
        }

        class_name = LAND_COVER_CLASSES.get(int(land_cover_value), "Unknown")

        return {
            "class_code": int(land_cover_value),
            "class_name": class_name
        }
    except Exception as e:
        print(f"Error fetching landcover data: {e}")
        return None

if __name__ == "__main__":
    data = get_landcover_data(19.0760, 72.8777)
    print("Environmental data:")
    print(data)