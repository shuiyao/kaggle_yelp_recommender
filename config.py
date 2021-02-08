# Global variables for content.py and als.py

FOLDER_INPUT = "./data/"

JSON_BUSINESS = FOLDER_INPUT + "business_pittsburgh.json"
CSV_BUSINESS = FOLDER_INPUT + "business_pittsburgh.csv"
CSV_REVIEW = FOLDER_INPUT + "review_pittsburgh.csv"
CSV_RATINGS = FOLDER_INPUT + "ratings_pittsburgh.csv"
CSV_USER = FOLDER_INPUT + "user_pittsburgh.csv"
CSV_CHECKIN = FOLDER_INPUT + "checkin_pittsburgh.csv"

CSV_TIP = FOLDER_INPUT + "tip_pittsburgh.csv"

Attributes_Boolean = ["Open24Hours", "HappyHour", "DogsAllowed", "WheelchairAccessible", "GoodForKids", "OutdoorSeating", "GoodForDancing", "GoodForGroups"]
Attributes_Cat = ["RestaurantsAttire", "NoiseLevel", "Wifi", "Alcohol", "Smoking", "RestaurantsPriceRange2"]
Attributes_Dict = ["Ambience", "BusinessParking"]
Attributes = Attributes_Boolean + Attributes_Cat

