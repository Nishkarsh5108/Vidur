"""
OpenStreetMap Overpass API Geofencing Engine with Offline Mock Support.

Queries real school POIs and polygons using the Overpass API, or utilizes built-in
synthetic Indian city school zone polygons (Bengaluru / Hyderabad) for offline development.
Provides fast Point-in-Polygon (PIP) testing using Shapely and OpenCV.
"""

import math
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import requests

try:
    from shapely.geometry import Point, Polygon
    from shapely.ops import transform
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GeofenceManager")


# Representative mock Overpass API data for Indian school zones (Bengaluru)
MOCK_OVERPASS_SCHOOL_DATA = {
    "version": 0.6,
    "generator": "Overpass API (Mocked for SIH 2026 Hackathon)",
    "elements": [
        {
            "type": "way",
            "id": 10101,
            "tags": {
                "amenity": "school",
                "name": "St. Joseph's Indian High School",
                "operator": "Jesuit",
                "city": "Bengaluru"
            },
            "center": {"lat": 12.9716, "lon": 77.5946},
            "geometry": [
                {"lat": 12.9720, "lon": 77.5940},
                {"lat": 12.9722, "lon": 77.5952},
                {"lat": 12.9712, "lon": 77.5954},
                {"lat": 12.9710, "lon": 77.5942},
                {"lat": 12.9720, "lon": 77.5940}
            ]
        },
        {
            "type": "way",
            "id": 10102,
            "tags": {
                "amenity": "school",
                "name": "Bishop Cotton Boys' School",
                "city": "Bengaluru"
            },
            "center": {"lat": 12.9698, "lon": 77.6015},
            "geometry": [
                {"lat": 12.9705, "lon": 77.6010},
                {"lat": 12.9706, "lon": 77.6022},
                {"lat": 12.9692, "lon": 77.6025},
                {"lat": 12.9690, "lon": 77.6012},
                {"lat": 12.9705, "lon": 77.6010}
            ]
        },
        {
            "type": "node",
            "id": 10103,
            "lat": 12.9750,
            "lon": 77.5980,
            "tags": {
                "amenity": "school",
                "name": "Kendriya Vidyalaya M.G. Road",
                "city": "Bengaluru"
            }
        }
    ]
}


@dataclass
class SchoolZone:
    id: int
    name: str
    center_lat: float
    center_lon: float
    buffer_meters: float
    polygon_coords: List[Tuple[float, float]]  # [(lat, lon), ...]
    polygon_obj: Any = None                   # Shapely Polygon instance


class SchoolGeofenceManager:
    """
    Manages school locations, geofence polygons, and coordinates testing.
    Can query live Overpass API or use bundled offline mock data.
    """
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"

    def __init__(self, city_name: str = "Bengaluru", buffer_meters: float = 120.0, use_mock: bool = True):
        self.city_name = city_name
        self.buffer_meters = buffer_meters
        self.use_mock = use_mock
        self.school_zones: List[SchoolZone] = []

        if not HAS_SHAPELY:
            logger.warning("Shapely is not installed. Falling back to ray-casting algorithm for PIP.")

        # Initialize zones
        if self.use_mock:
            self.load_mock_data()
        else:
            try:
                self.fetch_live_overpass(city_name=self.city_name)
            except Exception as e:
                logger.warning(f"Overpass live query failed ({e}). Falling back to mock data.")
                self.load_mock_data()

    def load_mock_data(self):
        """Loads bundled mock school zones for instant testing."""
        logger.info(f"Loading offline mock school geofences for {self.city_name}...")
        self.school_zones = self._parse_overpass_json(MOCK_OVERPASS_SCHOOL_DATA)
        logger.info(f"Loaded {len(self.school_zones)} mock school geofence zones.")

    def fetch_live_overpass(self, city_name: str, timeout: int = 25) -> List[SchoolZone]:
        """
        Executes an Overpass QL query to find all schools in a given Indian city.
        """
        logger.info(f"Querying live OpenStreetMap Overpass API for schools in '{city_name}'...")
        query = f"""
        [out:json][timeout:{timeout}];
        area[name="{city_name}"][admin_level~"4|6|8"]->.searchArea;
        (
          node["amenity"="school"](area.searchArea);
          way["amenity"="school"](area.searchArea);
          relation["amenity"="school"](area.searchArea);
        );
        out body geom;
        >;
        out skel qt;
        """

        response = requests.post(
            self.OVERPASS_URL,
            data={"data": query},
            headers={"User-Agent": "SIH2026-TrafficAnalytics/1.0"},
            timeout=timeout
        )

        if response.status_code != 200:
            raise RuntimeError(f"Overpass query returned status {response.status_code}")

        data = response.json()
        self.school_zones = self._parse_overpass_json(data)
        logger.info(f"Fetched {len(self.school_zones)} school geofences from Overpass API.")
        return self.school_zones

    def _meters_to_degrees(self, meters: float, lat: float) -> Tuple[float, float]:
        """Converts buffer distance in meters to approximate delta latitude and delta longitude."""
        delta_lat = meters / 111139.0
        delta_lon = meters / (111139.0 * math.cos(math.radians(lat)))
        return delta_lat, delta_lon

    def _parse_overpass_json(self, data: Dict[str, Any]) -> List[SchoolZone]:
        """Converts Overpass JSON elements into structured SchoolZone objects."""
        zones = []
        for elem in data.get("elements", []):
            tags = elem.get("tags", {})
            if tags.get("amenity") != "school":
                continue

            elem_id = elem.get("id", 0)
            name = tags.get("name", f"School #{elem_id}")

            # Extract coordinates
            if elem.get("type") == "way" and "geometry" in elem:
                coords = [(pt["lat"], pt["lon"]) for pt in elem["geometry"]]
                c_lat = sum(c[0] for c in coords) / len(coords)
                c_lon = sum(c[1] for c in coords) / len(coords)
            elif elem.get("type") == "node":
                c_lat = elem.get("lat", 0.0)
                c_lon = elem.get("lon", 0.0)
                # Expand node point into a square buffer polygon
                d_lat, d_lon = self._meters_to_degrees(self.buffer_meters, c_lat)
                coords = [
                    (c_lat + d_lat, c_lon - d_lon),
                    (c_lat + d_lat, c_lon + d_lon),
                    (c_lat - d_lat, c_lon + d_lon),
                    (c_lat - d_lat, c_lon - d_lon),
                    (c_lat + d_lat, c_lon - d_lon)
                ]
            else:
                continue

            # Create Shapely Polygon (lon, lat order in standard GIS)
            poly_obj = None
            if HAS_SHAPELY and len(coords) >= 3:
                try:
                    poly_obj = Polygon([(pt[1], pt[0]) for pt in coords])
                except Exception:
                    poly_obj = None

            zones.append(SchoolZone(
                id=elem_id,
                name=name,
                center_lat=c_lat,
                center_lon=c_lon,
                buffer_meters=self.buffer_meters,
                polygon_coords=coords,
                polygon_obj=poly_obj
            ))
        return zones

    def point_in_polygon_raycast(self, lat: float, lon: float, poly_coords: List[Tuple[float, float]]) -> bool:
        """Standard ray casting algorithm for point-in-polygon without external GIS libraries."""
        n = len(poly_coords)
        inside = False
        p1_lat, p1_lon = poly_coords[0]
        for i in range(1, n + 1):
            p2_lat, p2_lon = poly_coords[i % n]
            if lon > min(p1_lon, p2_lon):
                if lon <= max(p1_lon, p2_lon):
                    if lat <= max(p1_lat, p2_lat):
                        if p1_lon != p2_lon:
                            lat_inters = (lon - p1_lon) * (p2_lat - p1_lat) / (p2_lon - p1_lon) + p1_lat
                        if p1_lat == p2_lat or lat <= lat_inters:
                            inside = not inside
            p1_lat, p1_lon = p2_lat, p2_lon
        return inside

    def check_coordinates(self, lat: float, lon: float) -> Tuple[bool, Optional[str], Optional[SchoolZone]]:
        """
        Checks if given (lat, lon) falls inside any school geofence zone.
        Returns: (is_inside, school_name, zone_object)
        """
        if HAS_SHAPELY:
            pt = Point(lon, lat)
            for zone in self.school_zones:
                if zone.polygon_obj and zone.polygon_obj.contains(pt):
                    return True, zone.name, zone
        else:
            for zone in self.school_zones:
                if self.point_in_polygon_raycast(lat, lon, zone.polygon_coords):
                    return True, zone.name, zone

        return False, None, None


# Quick self-test when executed directly
if __name__ == "__main__":
    print("[*] Running standalone GeofenceManager verification...")
    mgr = SchoolGeofenceManager(city_name="Bengaluru", use_mock=True)

    # Test point 1: Inside St. Joseph's
    inside_lat, inside_lon = 12.9716, 77.5946
    in_zone, name, _ = mgr.check_coordinates(inside_lat, inside_lon)
    print(f"Test 1 (Inside School): Coord=({inside_lat}, {inside_lon}) -> Match={in_zone} ({name})")
    assert in_zone is True, "Failed: Point should be inside St. Joseph's"

    # Test point 2: Far outside (Electronic City)
    outside_lat, outside_lon = 12.8399, 77.6770
    out_zone, name, _ = mgr.check_coordinates(outside_lat, outside_lon)
    print(f"Test 2 (Outside School): Coord=({outside_lat}, {outside_lon}) -> Match={out_zone} ({name})")
    assert out_zone is False, "Failed: Point should be outside school zones"

    print("[+] GeofenceManager unit tests PASSED successfully!")
