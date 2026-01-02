from flask import Flask, request
import pickle

app = Flask(__name__)


model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return """
    <h2>SMS Spam Detection</h2>
    <form action="/predict" method="post">
        <textarea name="message" rows="5" cols="40"></textarea><br><br>
        <input type="submit" value="Predict">
    </form>
    """

@app.route("/predict", methods=["POST"])
def predict():
    msg = request.form["message"]
    data = vectorizer.transform([msg])
    pred = model.predict(data)[0]

    return "🚨 Spam Message" if pred == 1 else "✅ Not Spam (Ham)"

if __name__ == "__main__":
   
    app.run(debug=True, use_reloader=False)
