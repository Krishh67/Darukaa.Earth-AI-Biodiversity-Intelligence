import json
import math
from io import BytesIO

import numpy as np
import requests
import rasterio


# ============================================================
# USER LOCATION
# ============================================================

LAT = 19.076
LON = 72.8777


# ============================================================
# SEARCH AREA
#
# 0.15 degrees gives approximately:
# ~16 km north/south around Mumbai
#
# This is large enough to find nearby non-urban soil cells
# while keeping the WCS requests reasonably small.
# ============================================================

BOX_SIZE = 0.15

LAT_MIN = LAT - BOX_SIZE
LAT_MAX = LAT + BOX_SIZE

LON_MIN = LON - BOX_SIZE
LON_MAX = LON + BOX_SIZE


# ============================================================
# SOILGRIDS WCS
# ============================================================

BASE_URL = "https://maps.isric.org/mapserv"


# ============================================================
# SOILGRIDS VARIABLES
#
# All variables are taken from the 0-5 cm depth.
# ============================================================

SOIL_PROPERTIES = {

    "soc": {
        "map": "soc",
        "coverage": "soc_0-5cm_mean",
        "unit": "g/kg"
    },

    "ph": {
        "map": "phh2o",
        "coverage": "phh2o_0-5cm_mean",
        "unit": "pH"
    },

    "clay": {
        "map": "clay",
        "coverage": "clay_0-5cm_mean",
        "unit": "%"
    },

    "sand": {
        "map": "sand",
        "coverage": "sand_0-5cm_mean",
        "unit": "%"
    },

    "silt": {
        "map": "silt",
        "coverage": "silt_0-5cm_mean",
        "unit": "%"
    },

    "nitrogen": {
        "map": "nitrogen",
        "coverage": "nitrogen_0-5cm_mean",
        "unit": "g/kg"
    },

    "bulk_density": {
        "map": "bdod",
        "coverage": "bdod_0-5cm_mean",
        "unit": "kg/dm3"
    }
}


# ============================================================
# REQUEST ONE SOILGRIDS VARIABLE
# ============================================================

def request_soilgrids(property_name, config):

    map_name = config["map"]
    coverage_id = config["coverage"]

    params = [

        (
            "map",
            f"/map/{map_name}.map"
        ),

        ("SERVICE", "WCS"),

        ("VERSION", "2.0.1"),

        ("REQUEST", "GetCoverage"),

        ("COVERAGEID", coverage_id),

        ("FORMAT", "GEOTIFF_INT16"),

        # Geographic subset
        (
            "SUBSET",
            f"X({LON_MIN},{LON_MAX})"
        ),

        (
            "SUBSET",
            f"Y({LAT_MIN},{LAT_MAX})"
        ),

        # Input CRS
        (
            "SUBSETTINGCRS",
            "http://www.opengis.net/def/crs/EPSG/0/4326"
        ),

        # Output CRS
        (
            "OUTPUTCRS",
            "http://www.opengis.net/def/crs/EPSG/0/4326"
        )
    ]

    print(f"Requesting {property_name}...")

    try:

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=180
        )

    except requests.RequestException as e:

        print(
            f"Request failed for {property_name}: {e}"
        )

        return None

    print(
        f"  HTTP: {response.status_code}"
    )

    if response.status_code != 200:

        print(
            f"  SoilGrids error for {property_name}:"
        )

        print(
            response.text[:1000]
        )

        return None

    content_type = response.headers.get(
        "Content-Type",
        ""
    ).lower()

    if "tiff" not in content_type:

        print(
            f"  Unexpected response for {property_name}: "
            f"{content_type}"
        )

        return None

    return response.content


# ============================================================
# READ TIFF
# ============================================================

def read_tiff(tiff_data):

    dataset = rasterio.open(
        BytesIO(tiff_data)
    )

    data = dataset.read(1)

    return dataset, data


# ============================================================
# GET PIXEL INDEX FOR USER COORDINATE
# ============================================================

def get_pixel_index(dataset):

    row, col = dataset.index(
        LON,
        LAT
    )

    return row, col


# ============================================================
# HAVERSINE DISTANCE
#
# Returns distance in kilometres.
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    earth_radius_km = 6371.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    delta_lat = math.radians(
        lat2 - lat1
    )

    delta_lon = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(delta_lat / 2) ** 2
        +
        math.cos(lat1_rad)
        *
        math.cos(lat2_rad)
        *
        math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return earth_radius_km * c


# ============================================================
# PIXEL -> LAT/LON
# ============================================================

def pixel_to_coordinates(
    dataset,
    row,
    col
):

    lon, lat = dataset.xy(
        row,
        col,
        offset="center"
    )

    return float(lat), float(lon)


# ============================================================
# FIND NEAREST COMMON VALID PIXEL
#
# IMPORTANT:
#
# We don't independently find a nearby SOC pixel,
# nearby pH pixel, etc.
#
# Instead we find ONE pixel where ALL variables
# are available.
#
# This keeps all soil measurements spatially consistent.
# ============================================================

def find_nearest_common_valid_pixel(
    datasets,
    arrays
):

    reference_dataset = datasets["soc"]

    # --------------------------------------------------------
    # Start from the user's exact pixel
    # --------------------------------------------------------

    center_row, center_col = get_pixel_index(
        reference_dataset
    )

    height = reference_dataset.height
    width = reference_dataset.width

    # --------------------------------------------------------
    # Create common validity mask
    #
    # Every requested soil property must have a positive
    # value at the same pixel.
    # --------------------------------------------------------

    common_valid = np.ones(
        (height, width),
        dtype=bool
    )

    for property_name, data in arrays.items():

        common_valid &= (
            data > 0
        )

    # --------------------------------------------------------
    # Check exact pixel first
    # --------------------------------------------------------

    if (
        0 <= center_row < height
        and
        0 <= center_col < width
        and
        common_valid[
            center_row,
            center_col
        ]
    ):

        return (
            center_row,
            center_col,
            0.0,
            "exact"
        )

    # --------------------------------------------------------
    # Find every common valid pixel
    # --------------------------------------------------------

    valid_rows, valid_cols = np.where(
        common_valid
    )

    if len(valid_rows) == 0:

        return None

    # --------------------------------------------------------
    # Find closest valid pixel using geographic distance
    # --------------------------------------------------------

    best_distance = float("inf")
    best_row = None
    best_col = None

    for row, col in zip(
        valid_rows,
        valid_cols
    ):

        candidate_lat, candidate_lon = (
            pixel_to_coordinates(
                reference_dataset,
                int(row),
                int(col)
            )
        )

        distance = haversine_distance(
            LAT,
            LON,
            candidate_lat,
            candidate_lon
        )

        if distance < best_distance:

            best_distance = distance

            best_row = int(row)
            best_col = int(col)

    if best_row is None:

        return None

    return (
        best_row,
        best_col,
        best_distance,
        "nearby"
    )


# ============================================================
# CONVERSION
# ============================================================

def convert_value(
    property_name,
    raw_value
):

    if raw_value is None:

        return None

    # --------------------------------------------------------
    # SOC
    #
    # SoilGrids storage:
    # dg/kg
    #
    # Output:
    # g/kg
    # --------------------------------------------------------

    if property_name == "soc":

        return round(
            float(raw_value) / 10.0,
            3
        )

    # --------------------------------------------------------
    # pH
    #
    # Stored as pH * 10
    # --------------------------------------------------------

    if property_name == "ph":

        return round(
            float(raw_value) / 10.0,
            2
        )

    # --------------------------------------------------------
    # Clay
    # Sand
    # Silt
    #
    # Stored as g/kg
    #
    # Convert to percentage
    # --------------------------------------------------------

    if property_name in [
        "clay",
        "sand",
        "silt"
    ]:

        return round(
            float(raw_value) / 10.0,
            2
        )

    # --------------------------------------------------------
    # Nitrogen
    #
    # cg/kg -> g/kg
    # --------------------------------------------------------

    if property_name == "nitrogen":

        return round(
            float(raw_value) / 100.0,
            3
        )

    # --------------------------------------------------------
    # Bulk density
    #
    # cg/cm3 -> kg/dm3
    # --------------------------------------------------------

    if property_name == "bulk_density":

        return round(
            float(raw_value) / 100.0,
            3
        )

    return float(raw_value)


# ============================================================
# MAIN SOILGRIDS FUNCTION
# ============================================================

def get_soil_data(
    latitude,
    longitude
):

    global LAT
    global LON

    global LAT_MIN
    global LAT_MAX

    global LON_MIN
    global LON_MAX

    # --------------------------------------------------------
    # Update coordinates
    # --------------------------------------------------------

    LAT = latitude
    LON = longitude

    LAT_MIN = LAT - BOX_SIZE
    LAT_MAX = LAT + BOX_SIZE

    LON_MIN = LON - BOX_SIZE
    LON_MAX = LON + BOX_SIZE

    print("\n")
    print("=" * 70)
    print("SOILGRIDS ENVIRONMENTAL DATA")
    print("=" * 70)

    print(
        f"\nRequested location:"
    )

    print(
        f"Latitude : {LAT}"
    )

    print(
        f"Longitude: {LON}"
    )

    print(
        f"\nSearch area: "
        f"{LAT_MIN:.4f} → {LAT_MAX:.4f} latitude"
    )

    print(
        f"             "
        f"{LON_MIN:.4f} → {LON_MAX:.4f} longitude"
    )

    # --------------------------------------------------------
    # Download all SoilGrids layers
    # --------------------------------------------------------

    datasets = {}
    arrays = {}

    for property_name, config in SOIL_PROPERTIES.items():

        print("\n" + "-" * 70)

        tiff_data = request_soilgrids(
            property_name,
            config
        )

        if tiff_data is None:

            print(
                f"Could not retrieve {property_name}"
            )

            return {
                "location": {
                    "latitude": LAT,
                    "longitude": LON
                },
                "soil": None,
                "availability": "error",
                "source": "SoilGrids"
            }

        try:

            dataset, data = read_tiff(
                tiff_data
            )

        except Exception as e:

            print(
                f"Could not read {property_name}: {e}"
            )

            return {
                "location": {
                    "latitude": LAT,
                    "longitude": LON
                },
                "soil": None,
                "availability": "error",
                "source": "SoilGrids"
            }

        datasets[property_name] = dataset
        arrays[property_name] = data

        print(
            f"  Raster: "
            f"{dataset.width} x {dataset.height}"
        )

        print(
            f"  Range: "
            f"{np.min(data)} → {np.max(data)}"
        )

    # --------------------------------------------------------
    # Check that all rasters have the same dimensions
    # --------------------------------------------------------

    reference = datasets["soc"]

    for property_name, dataset in datasets.items():

        if (
            dataset.width != reference.width
            or
            dataset.height != reference.height
        ):

            print(
                f"Raster dimension mismatch: "
                f"{property_name}"
            )

            return {
                "location": {
                    "latitude": LAT,
                    "longitude": LON
                },
                "soil": None,
                "availability": "error",
                "source": "SoilGrids"
            }

    # --------------------------------------------------------
    # Find nearest common valid pixel
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SEARCHING FOR VALID SOIL LOCATION")
    print("=" * 70)

    result = find_nearest_common_valid_pixel(
        datasets,
        arrays
    )

    if result is None:

        print(
            "\nNo common valid SoilGrids pixel "
            "found in search area."
        )

        output = {

            "location": {
                "latitude": LAT,
                "longitude": LON
            },

            "soil": {
                "soc_g_kg": None,
                "ph": None,
                "clay_percent": None,
                "sand_percent": None,
                "silt_percent": None,
                "nitrogen_g_kg": None,
                "bulk_density_kg_dm3": None
            },

            "availability": "unavailable",

            "source": "SoilGrids",

            "depth": "0-5 cm"
        }

        return output

    (
        row,
        col,
        distance_km,
        status
    ) = result

    # --------------------------------------------------------
    # Coordinates of actual soil pixel
    # --------------------------------------------------------

    soil_lat, soil_lon = (
        pixel_to_coordinates(
            reference,
            row,
            col
        )
    )

    print(
        "\nValid soil pixel found."
    )

    print(
        f"Pixel row     : {row}"
    )

    print(
        f"Pixel column  : {col}"
    )

    print(
        f"Soil latitude : {soil_lat}"
    )

    print(
        f"Soil longitude: {soil_lon}"
    )

    print(
        f"Distance      : {distance_km:.3f} km"
    )

    print(
        f"Status        : {status}"
    )

    # --------------------------------------------------------
    # Extract and convert all values
    # --------------------------------------------------------

    soil = {}

    for property_name, config in SOIL_PROPERTIES.items():

        raw_value = arrays[
            property_name
        ][
            row,
            col
        ]

        converted = convert_value(
            property_name,
            raw_value
        )

        if property_name == "soc":

            soil["soc_g_kg"] = converted

        elif property_name == "ph":

            soil["ph"] = converted

        elif property_name == "clay":

            soil["clay_percent"] = converted

        elif property_name == "sand":

            soil["sand_percent"] = converted

        elif property_name == "silt":

            soil["silt_percent"] = converted

        elif property_name == "nitrogen":

            soil["nitrogen_g_kg"] = converted

        elif property_name == "bulk_density":

            soil["bulk_density_kg_dm3"] = converted

    # --------------------------------------------------------
    # Final JSON-compatible output
    # --------------------------------------------------------

    output = {

        "location": {

            "latitude": LAT,

            "longitude": LON

        },

        "soil_location": {

            "latitude": round(
                soil_lat,
                6
            ),

            "longitude": round(
                soil_lon,
                6
            ),

            "distance_from_requested_location_km":
                round(
                    distance_km,
                    3
                )

        },

        "soil": soil,

        "availability": status,

        "source": "SoilGrids",

        "depth": "0-5 cm"

    }

    return output


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    data,
    filename="soilgrids_result.json"
):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )

    print(
        f"\nSaved result to: {filename}"
    )


# ============================================================
# PRINT FINAL RESULT
# ============================================================

def print_result(data):

    print("\n")
    print("=" * 70)
    print("FINAL SOILGRIDS RESULT")
    print("=" * 70)

    print(
        json.dumps(
            data,
            indent=4
        )
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    result = get_soil_data(
        LAT,
        LON
    )

    print_result(
        result
    )

    save_json(
        result
    )