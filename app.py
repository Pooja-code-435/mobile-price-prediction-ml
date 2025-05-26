from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

# step 2:loading the saved model
with open("place-scaler-model.pkl","rb")as file:
    place_model=pickle.load(file)
    
# step 2:loading the saved model
with open("mobile-rf-model.pkl","rb")as file:
    price_model=pickle.load(file)
    
    
def predict_phone_quality(performance=8.5, storage_capacity=128, camera_quality=9.0, battery_life=24, weight=180, age=1.0):
    # Prepare feature list in the correct order
    features = [
        float(performance),
        int(storage_capacity),
        float(camera_quality),
        float(battery_life),
        float(weight),
        float(age)
    ]

    # Ensure the correct number of features
    if len(features) != 6:
        print("Error: Feature list does not have 6 elements.")
        print("Current features:", features)
        return

    # Convert to 2D array for scaler
    features_array = np.array([features])

    # Apply standard scaling
    features_scaled = place_model.transform(features_array)

    # Predict using the scaled input
    result = price_model.predict(features_scaled)
    print("Raw model output:", result)
    return result[0]
    

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/contact", methods=["GET"])
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        performance=request.form.get("performance")
        storage_capacity=request.form.get("storage_capacity")
        camera_quality=request.form.get("camera_quality")
        battery_life=request.form.get("battery_life")
        weight=request.form.get("weight")
        age=request.form.get("age")
        # Make prediction
        result = predict_phone_quality(
            performance=performance,
            storage_capacity=storage_capacity,
            camera_quality=camera_quality,
            battery_life=battery_life,
            weight=weight,
            age=age
        )
        print(result)
        # Render result
        return render_template("predict.html", prediction=result)

    return render_template("predict.html")

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)