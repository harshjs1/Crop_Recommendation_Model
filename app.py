from flask import Flask, request, render_template
import numpy as np
import pickle
import logging
import os
# Initialize the Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load the models and scalers
try:
    model = pickle.load(open('model/modelrandclf.pkl', 'rb'))
    sc = pickle.load(open('model/standscaler.pkl', 'rb'))
    mx = pickle.load(open('model/minmaxscaler.pkl', 'rb'))
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or scalers: {e}")
    raise

# Route to the home page
@app.route('/')
def index():
    print("Current working directory:", os.getcwd())
    return render_template("index.html")

# Route for predictions
@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Retrieve form data
        N = request.form.get('Nitrogen')
        P = request.form.get('Phosporus')
        K = request.form.get('Potassium')
        temp = request.form.get('Temperature')
        humidity = request.form.get('Humidity')
        ph = request.form.get('pH')
        rainfall = request.form.get('Rainfall')

        # Validate inputs
        if not all([N, P, K, temp, humidity, ph, rainfall]):
            return render_template("index.html", result="Please fill in all fields.")

        # Prepare the input for the model
        feature_list = [float(N), float(P), float(K), float(temp), float(humidity), float(ph), float(rainfall)]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Scale the input data
        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)

        # Make prediction
        prediction = model.predict(sc_mx_features)

        # Map prediction to crop names
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
            6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
            11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 
            15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        # Generate result message
        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = f"{crop} is the best crop to be cultivated right there."
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

        return render_template('index.html', result=result)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template("index.html", result="An error occurred during prediction. Please try again.")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
