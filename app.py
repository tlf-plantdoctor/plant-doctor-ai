import os
import replicate
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load your Replicate API token from environment
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Replace this with your actual model
model = replicate_client.models.get("nohamoamary/nabtah-plant-disease")

@app.route("/", methods=["GET"])
def home():
    return "Plant Doctor AI is running 🌱"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_url = data.get("image_url")

        if not image_url:
            return jsonify({"error": "No image URL provided"}), 400

        prediction = model.predict(image=image_url)

        return jsonify({
            "status": "success",
            "result": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
