from flask import Flask,jsonify,request
import numpy as np
import pandas as pd
import joblib


app = Flask(__name__)

# with open("new_pipe_price_predictor22.pkl", "rb") as f:
#     price_model = joblib.load(f)
#
# with open("new_pipeRating2.pkl", "rb") as f:
#     rating_model = joblib.load(f)

with open("compressed_crop_produciton_model.pkl", "rb") as f:
    production = joblib.load(f)

with open("compressed_crop_recommender_model.pkl", "rb") as f:
    cropRecommender = joblib.load(f)

with open("compressed_fertilser_model.pkl", "rb") as f:
    fertiliser= joblib.load(f)

with open("compressed_rainfall_model.pkl", "rb") as f:
    rainfall = joblib.load(f)


expected_columns = [
    "state_name", "district_name", "market_name", "commodity_name",
    "variety", "grade", "year", "month", "day"
]
expected_columns2= [
    "state_name", "district_name", "market_name", "commodity_name",
    "variety", "grade","modal_price", "year", "month", "day"
]
# Define expected data types for validation
expected_columns3 = ["State_Name", "District_Name", "Crop_Year", "Season" ,"Crop", "Area"]

expected_columns4=["N","P","K","temperature","humidity","ph","rainfall"]

expected_columns5=["Crop", "pH"]

expected_columns6=["STATE_UT_NAME", "DISTRICT", "lat", "longs"]
def map_grade_to_score(df):
    grade_weights = {
        '1': 10,
        '2': 6,
        '3': 3
    }
    df['grade_score'] = df['grade'].map(grade_weights).astype(float)
    return df
    


# @app.route('/api/price-predictor', methods=['POST'])
# def price_predictor():
#     try:
#         # Parse the JSON input into a DataFrame
#         data = request.get_json()
#         print(data)
#         input_data = pd.DataFrame(data, columns=expected_columns)
#         print(input_data)
#         # Ensure the data has the correct format and types
#         # Ensure 'grade' is a string
#         input_data["grade"] = input_data["grade"].astype(str)
#         input_data["year"] = input_data["year"].astype(int)
#         input_data["month"] = input_data["month"].astype(int)
#         input_data["day"] = input_data["day"].astype(int)
#
#         # Perform prediction using the model
#         predictions = price_model.predict(input_data.iloc[[0]])+200
#
#         # Return predictions as a JSON response
#         return jsonify({"predicted_prices": predictions.tolist()})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400


# API to recommend ratings
# @app.route('/api/recommend-rating', methods=['POST'])
# def recommend_rating():
#     try:
#         # Parse the JSON input into a DataFrame
#         data = request.get_json()
#         input_data = pd.DataFrame(data, columns=expected_columns2)
#
#         # Ensure correct data types
#         input_data["grade"] = input_data["grade"].astype(str)
#         input_data["modal_price"] = input_data["modal_price"].astype(float)
#         input_data["year"] = input_data["year"].astype(int)
#         input_data["month"] = input_data["month"].astype(int)
#         input_data["day"] = input_data["day"].astype(int)
#
#         # Apply transformations
#         input_data = map_grade_to_score(input_data)
#         print(input_data)
#         # Feed transformed data into the model for prediction
#         predictions = rating_model.predict(input_data.iloc[[0]])
#
#         # Return predictions as a JSON response
#         return jsonify({"recommended_ratings": predictions.tolist()})
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400


@app.route('/api/crop-production', methods=['POST'])
def crop_production():
    try:
        # Get JSON data from the request
        data = request.get_json()
        input_data = pd.DataFrame(data, columns=expected_columns3)
        print(input_data)
        # Ensure correct data types
        input_data["State_Name"] = input_data["State_Name"].astype(str)
        input_data["District_Name"] = input_data["District_Name"].astype(str)
        input_data["Crop_Year"] = input_data["Crop_Year"].astype(int)
        input_data["Season"] = input_data["Season"].astype(str)
        input_data["Crop"] = input_data["Crop"].astype(str)
        input_data["Area"] = input_data["Area"].astype(float)

        # Make prediction
        prediction = production.predict(input_data.iloc[[0]])

        # Return prediction as JSON
        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/crop-recommender', methods=['POST'])
def crop_recommendation():
    try:
        label_to_crop = {
            0: "apple", 1: "banana", 2: "blackgram", 3: "chickpea", 4: "coconut",
            5: "coffee", 6: "cotton", 7: "grapes", 8: "jute", 9: "kidneybeans",
            10: "lentil", 11: "maize", 12: "mango", 13: "mothbeans", 14: "mungbean",
            15: "muskmelon", 16: "orange", 17: "papaya", 18: "pigeonpeas",
            19: "pomegranate", 20: "rice", 21: "watermelon"
        }
        # Get JSON data from the request
        data = request.get_json()

        input_data = pd.DataFrame(data, columns=expected_columns4)
        print(input_data)
        # Ensure correct data types
        input_data["N"] = input_data["N"].astype(int)
        input_data["P"] = input_data["P"].astype(int)
        input_data["K"] = input_data["K"].astype(int)
        input_data["temperature"] = input_data["temperature"].astype(float)
        input_data["humidity"] = input_data["humidity"].astype(float)
        input_data["ph"] = input_data["ph"].astype(float)
        input_data["rainfall"]=input_data["rainfall"].astype(float)

        # Make prediction
        prediction = cropRecommender.predict(input_data.iloc[[0]])
        # Convert the prediction to a native Python type
        prediction_native = prediction[0].item() if hasattr(prediction[0], "item") else prediction[0]
        crop_name = label_to_crop.get(prediction_native, "Unknown Crop")
        # Return prediction as JSON
        return jsonify({"prediction": crop_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/fertiliser', methods=['POST'])
def fertiliser_recommendation():
    try:
        # Get JSON data from the request
        data = request.get_json()
        input_data = pd.DataFrame(data, columns=expected_columns5)
        # Ensure correct data types
        input_data["Crop"] = input_data["Crop"].astype(str)
        input_data["pH"] = input_data["pH"].astype(float)

        # Make prediction
        prediction = fertiliser.predict(input_data.iloc[[0]])

        # Convert prediction to a native Python type
        # Convert prediction to a dictionary with keys N, P, K
        if isinstance(prediction, np.ndarray) and prediction.size == 3:
            prediction_dict = {
                "N": float(prediction[0][0]),
                "P": float(prediction[0][1]),
                "K": float(prediction[0][2])
            }
        else:
            return jsonify({"error": "Unexpected prediction format"}), 500

        # Return predictions as JSON
        return jsonify({"prediction": prediction_dict})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/rainfall', methods=['POST'])
def rainfall_twelve_months():
    try:
        # Get JSON data from the request
        data = request.get_json()
        input_data = pd.DataFrame(data, columns=expected_columns6)
        # Ensure correct data types
        input_data["STATE_UT_NAME"] = input_data["STATE_UT_NAME"].astype(str)
        input_data["DISTRICT"] = input_data["DISTRICT"].astype(str)
        input_data["lat"] = input_data["lat"].astype(float)
        input_data["longs"] = input_data["longs"].astype(float)
        # Make prediction
        prediction = rainfall.predict(input_data.iloc[[0]])
        if isinstance(prediction, np.ndarray) and prediction.shape == (1, 12):
            prediction_values = prediction[0]  # Get the first row of the array, which is a 1D array

            # Map the prediction to months
            prediction_dict = {
                "January": float(prediction_values[0]),
                "February": float(prediction_values[1]),
                "March": float(prediction_values[2]),
                "April": float(prediction_values[3]),
                "May": float(prediction_values[4]),
                "June": float(prediction_values[5]),
                "July": float(prediction_values[6]),
                "August": float(prediction_values[7]),
                "September": float(prediction_values[8]),
                "October": float(prediction_values[9]),
                "November": float(prediction_values[10]),
                "December": float(prediction_values[11])
            }
        else:
            return jsonify({"error": "Unexpected prediction format"}), 500

        # Return the monthly rainfall prediction as JSON
        return jsonify({"prediction": prediction_dict})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)