from pystac_client import Client
import planetary_computer
import rasterio


# ============================================================
# USER LOCATION
# ============================================================

LATITUDE = 19.0760
LONGITUDE = 72.8777


# ============================================================
# PLANETARY COMPUTER STAC API
# ============================================================

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

catalog = Client.open(
    CATALOG_URL,
    modifier=planetary_computer.sign_inplace
)


# ============================================================
# SEARCH ESA WORLDCOVER
# ============================================================

search = catalog.search(
    collections=["esa-worldcover"],
    intersects={
        "type": "Point",
        "coordinates": [LONGITUDE, LATITUDE]
    }
)

items = list(search.items())

if not items:
    raise RuntimeError("No ESA WorldCover data found for this location.")


item = items[0]

print("Found item:")
print(item.id)


# ============================================================
# FIND THE LAND-COVER ASSET
# ============================================================

print("\nAvailable assets:")

for key in item.assets:
    print(" -", key)


# WorldCover normally contains the land-cover raster
# under the 'map' asset.
asset = item.assets.get("map")

if asset is None:
    raise RuntimeError(
        "Could not find the WorldCover 'map' asset."
    )


# ============================================================
# READ RASTER
# ============================================================

with rasterio.open(asset.href) as src:

    # Convert geographic coordinate to raster coordinate
    row, col = src.index(
        LONGITUDE,
        LATITUDE
    )

    # Read the pixel
    land_cover_value = src.read(1)[row, col]


# ============================================================
# ESA WORLDCOVER CLASS MAPPING
# ============================================================

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


class_name = LAND_COVER_CLASSES.get(
    int(land_cover_value),
    "Unknown"
)


# ============================================================
# OUTPUT
# ============================================================

environmental_data = {
    "location": {
        "latitude": LATITUDE,
        "longitude": LONGITUDE
    },

    "land_cover": {
        "class_code": int(land_cover_value),
        "class_name": class_name
    }
}


print("\nEnvironmental data:")
print(environmental_data)