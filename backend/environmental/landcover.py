from pystac_client import Client
import planetary_computer
import rasterio

def get_landcover_data(lat: float, lon: float):
    """
    Fetches ESA WorldCover data from Microsoft Planetary Computer.
    """
    try:
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace
        )

        search = catalog.search(
            collections=["esa-worldcover"],
            intersects={
                "type": "Point",
                "coordinates": [lon, lat]
            }
        )

        items = list(search.items())
        if not items:
            return None

        item = items[0]
        asset = item.assets.get("map")
        if asset is None:
            return None

        with rasterio.open(asset.href) as src:
            row, col = src.index(lon, lat)
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

