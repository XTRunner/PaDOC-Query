import csv
from geopy.geocoders import Nominatim, ArcGIS
from geopy.exc import GeocoderTimedOut

city = 'New York, USA'


def get_lng_lat(geolocator, address):
    try:
        return geolocator.geocode(address, timeout=10)
    except GeocoderTimedOut:
        return get_lng_lat(geolocator, address)


def main():
    places = {}

    with open("tripAdvisorCrawler/attractionNYAddress.csv", "r", encoding='utf-8') as rhandle:
        rfile = csv.reader(rhandle)

        fields = next(rfile)

        poiIdx, addIdx = fields.index("place"), fields.index("address")

        for each_row in rfile:
            if len(each_row[addIdx]) >= 2:
                place = ", ".join(each_row)
            else:
                place = each_row[poiIdx] + ", " + city

            places[each_row[poiIdx]] = place

    # print(len(places))

    res = {}
    osm_geolocator = Nominatim(user_agent="tester")
    arcgis_geolocator = ArcGIS()

    for k, v in places.items():
        print(k.encode("utf-8"))
        print(v.encode("utf-8"))

        location = get_lng_lat(osm_geolocator, v.encode("utf-8"))
        if location:
            res[k] = (location.longitude, location.latitude)
        else:
            location = get_lng_lat(arcgis_geolocator, v.encode("utf-8"))
            if location:
                res[k] = (location.longitude, location.latitude)
            else:
                print("Not Found")

        print(res[k])
        print("Done: ", len(res), " / ", len(places))
        print("-------------------------------------")

    #pickle.dump(res, open("attractionNYLngLat.pkl", "wb"))

    with open("attractionNYLngLat.csv", 'a', newline='') as whandle:
        spamwriter = csv.writer(whandle)

        spamwriter.writerow(['place', 'lng', 'lat'])

        for k, v in res.items():
            spamwriter.writerow([k, v[0], v[1]])


if __name__ == '__main__':
    main()