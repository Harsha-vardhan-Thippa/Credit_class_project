from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("credit_card.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Decision threshold (IMPORTANT)
THRESHOLD = 0.5

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":

        # ---------- Numeric Inputs ----------
        age = float(request.form["age"])
        income = float(request.form["income"])
        open_loans = float(request.form["open_loans"])
        real_estate = float(request.form["real_estate"])

        # ---------- Gender (One-Hot) ----------
        Gender_Male = 1 if request.form["gender"] == "Male" else 0

        # ---------- Region (One-Hot, Central dropped) ----------
        region = request.form["region"]
        Region_East  = 1 if region == "East" else 0
        Region_North = 1 if region == "North" else 0
        Region_South = 1 if region == "South" else 0
        Region_West  = 1 if region == "West" else 0

        # ---------- Ordinal Encoding ----------
        house_map = {"Rented": 0, "Ownhouse": 1}
        occupation_map = {
            "Non-officer": 1,
            "Officer1": 2,
            "Officer2": 3,
            "Officer3": 4,
            "Self_Emp": 5
        }
        education_map = {
            "Matric": 1,
            "Graduate": 2,
            "Professional": 3,
            "Post-Grad": 4,
            "PhD": 5
        }

        Rented_OwnHouse_ordinal = house_map[request.form["house"]]
        Occupation_ordinal = occupation_map[request.form["occupation"]]
        Education_ordinal = education_map[request.form["education"]]

        # ---------- Final Input Vector (12 FEATURES ONLY) ----------
        input_data = np.array([[
            age,
            income,
            open_loans,
            real_estate,
            Gender_Male,
            Region_East,
            Region_North,
            Region_South,
            Region_West,
            Rented_OwnHouse_ordinal,
            Occupation_ordinal,
            Education_ordinal
        ]])

        # ---------- Scale ----------
        input_scaled = scaler.transform(input_data)

        # ---------- Predict ----------
        prob = model.predict_proba(input_scaled)[0][1]

        # ---------- Decision ----------
        if prob >= THRESHOLD:
            prediction = f"Approved ✅ (Score: {prob:.2f})"
        else:
            prediction = f"Rejected ❌ (Score: {prob:.2f})"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
