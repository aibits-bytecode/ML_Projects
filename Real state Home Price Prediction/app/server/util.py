import json
import pickle

import numpy as np

__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())

    except:
        raise Exception('Invalid location')

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index > 0:
        x[loc_index] = 1
    global __model
    return round(__model.predict([x])[0], 2)


def get_location_names():
    return __locations


def load_saved_artifacts():
    print("loading the saved artifacts ....")
    global __data_columns
    global __locations

    with open('./artifacts/columns.json', 'r') as f1:
        __data_columns = json.load(f1)['data_columns']
        __locations = __data_columns[3:]

    global __model
    with open('./artifacts/house_price_prediction_linear_model.pickle', 'rb') as f2:
        __model = pickle.load(f2)
        print("model loaded")

    print("loading the artifacts is done ...... ")


if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
