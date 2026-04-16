from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    day = request.form.get("day")

    if not day:
        return render_template("index.html", result="Enter a value")

    day = int(day)

    prediction = model.predict([[day]])
    prediction = round(prediction[0][0], 2)

    return render_template("index.html", result=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
