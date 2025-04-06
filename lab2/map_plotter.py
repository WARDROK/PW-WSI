import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

# Lista miast
cities = [
    'Ciechanów', 'Ostrołęka', 'Rzeszów', 'Zamość', 'Lublin', 'Chełm', 'Krosno',
    'Zakopane', 'Kraków', 'Cieszyn', 'Nowy Sącz', 'Bielsko-Biała', 'Przemyśl',
    'Katowice', 'Jelenia Góra', 'Legnica', 'Wałbrzych', 'Leszno', 'Słupsk',
    'Szczecin', 'Koszalin', 'Bydgoszcz', 'Piła', 'Zielona Góra', 'Gorzów Wielkopolski',
    'Opole', 'Wrocław', 'Łódź', 'Płock', 'Toruń', 'Elbląg', 'Gdańsk', 'Włocławek',
    'Sieradz', 'Kielce', 'Biała Podlaska', 'Skierniewice', 'Poznań', 'Konin',
    'Częstochowa', 'Kalisz', 'Białystok', 'Siedlce', 'Radom', 'Tarnobrzeg',
    'Piotrków Trybunalski', 'Warszawa', 'Tarnów', 'Łomża', 'Suwałki', 'Olsztyn'
]

# Inicjalizacja geokodera
geolocator = Nominatim(user_agent="city_mapper")


# Funkcja do geokodowania z obsługą timeoutu
def geocode_city(city):
    try:
        location = geolocator.geocode(city + ", Poland")
        return (location.latitude, location.longitude) if location else None
    except GeocoderTimedOut:
        time.sleep(1)
        return geocode_city(city)


# Geokodowanie miast
city_coords = []
for city in cities:
    coords = geocode_city(city)
    if coords:
        city_coords.append((city, coords))

# Tworzenie mapy z punktem początkowym w Polsce
mapa = folium.Map(location=[52.0, 19.0], zoom_start=6)

# Dodanie miast do mapy
for city, coords in city_coords:
    folium.Marker(
        location=coords,
        popup=city,
        icon=folium.Icon(color='blue')
    ).add_to(mapa)

# Dodanie linii łączących miasta
folium.PolyLine(
    locations=[coords for city, coords in city_coords],
    color='red',
    weight=3,
    opacity=0.7
).add_to(mapa)

# Zapisanie mapy do pliku HTML
mapa.save("polska_mapa_miast.html")

# Wyświetlenie mapy w Jupyter Notebook (jeśli używasz)
mapa
