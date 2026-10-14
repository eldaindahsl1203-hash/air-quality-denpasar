import geopandas as gpd

BASE_FOLD = "C:/Users/VICTUS/airflow/air-quality-denpasar"
path = BASE_FOLD + "/data/boundary/denpasar.shp"

gdf = gpd.read_file(path)

print("CRS:")
print(gdf.crs)

print("\nJumlah polygon:")
print(len(gdf))

print("\nKolom:")
print(gdf.columns.tolist())

print("\nGeometry:")
print(gdf.geometry)