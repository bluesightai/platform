import geopandas as gpd
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, GeoPoint, GeoRadius, ScoredPoint
from shapely.wkt import loads

url: str = "https://qdrant.malevich.ai/"
client = QdrantClient(url=url, port=443, timeout=10)


def search_in_radius(lon: float, lat: float, index: str, radius: float = 2000, limit: int = 100):
    filter = FieldCondition(
        key="location",
        geo_radius=GeoRadius(
            center=GeoPoint(
                lon=lon,
                lat=lat,
            ),
            radius=radius,
        ),
    )
    matched = client.scroll(index, with_vectors=True, scroll_filter=Filter(must=[filter]))[0][0]
    print(matched)
    return matched


def recommend_in_radius(base_point_id: str, index: str, count: int, radius: float = 100):
    filter = FieldCondition(
        key="location",
        geo_radius=GeoRadius(
            center=GeoPoint(
                lon=lon,
                lat=lat,
            ),
            radius=radius,
        ),
    )
    return client.search(index, query_vector=base_point_id, limit=count, query_filter=Filter(must=[filter]))


def recommend_similar_in_radius(
    lon: float, lat: float, index: str, count: int = 10, inner_radius: float = 100, outer_radius: float = 1000
):
    matched = search_in_radius(lon, lat, index_name, radius)
    if not matched:
        return []
    matched = matched[0]
    return recommend_in_radius(matched.id, index, count, radius=outer_radius)


def parse_qdrant_to_gdf(points: list[ScoredPoint]) -> gpd.GeoDataFrame:
    data = {"id": [], "version": [], "score": [], "lat": [], "lon": [], "geometry": []}

    points = [point.model_dump() for point in points]

    for point in points:
        data["id"].append(point["id"])
        data["version"].append(point["version"])
        data["score"].append(point["score"])
        data["lat"].append(point["payload"]["location"]["lat"])
        data["lon"].append(point["payload"]["location"]["lon"])
        data["geometry"].append(loads(point["payload"]["geometry"]))

    gdf = gpd.GeoDataFrame(data, geometry="geometry")
    return gdf


if __name__ == "__main__":

    # index_name = "greenvision_military_bases"
    index_name = "greenvision_military_bases_dakota"

    # Coordinates of the point we are looking similarity for

    # lat = 48.41497464795346
    # lon = -101.34594859891416

    # dakota
    # lat, lon = 48.41497464795346, -101.34594859891416
    lat, lon = 48.41492212027302, -101.34612074483346
    # lat, lon = 48.42512435620207, -101.34615913396736
    # lat, lon = 36.23890558133199, -115.0237089421788
    # lat, lon = 48.415827205799935, -101.34786264130538

    # Can be constant
    radius = 0
    outer_radius = 10000000
    # count = 20000

    count = 20000

    matched = search_in_radius(lon, lat, index_name)
    recommended = recommend_in_radius(matched.vector, index_name, count=count, radius=outer_radius)

    parsed = parse_qdrant_to_gdf(points=recommended)

    parsed.to_file("./search_greenvision_military_bases_dakota.json", driver="GeoJSON")
