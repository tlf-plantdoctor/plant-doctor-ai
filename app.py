import replicate
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Set your Replicate API token here
os.environ["REPLICATE_API_TOKEN"] = "export REPLICATE_API_TOKEN=r8_U6WgJKEui5JIFsGCaOZKmQh7iSeH50121R96t"

# Initialize model
model = replicate.models.get("microsoft/beit-large-patch16-224-pt22k-ft22k")

@app.route("/")
def home():
    return "Plant Doctor AI is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    image_url = data.get("image_url")
    lang = data.get("lang", "en")  # default to English

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    try:
        output = model.predict(image=image_url)
        class_label = output  # might return string like "leaf spot"

        # Basic instructions (can be improved later)
        tips = {
            "leaf spot": {
                "en": "Your plant has Leaf Spot. Remove affected leaves and avoid overhead watering.",
                "ar": "نباتك مصاب ببقع الأوراق. قم بإزالة الأوراق المصابة وتجنب سقي الأوراق مباشرة."
            },
            "powdery mildew": {
                "en": "Powdery Mildew detected. Improve airflow and use a natural fungicide.",
                "ar": "تم اكتشاف البياض الدقيقي. حسّن التهوية واستخدم مبيدًا فطريًا طبيعيًا."
            }
        }

        diagnosis = tips.get(class_label.lower(), {
            "en": f"Detected issue: {class_label}. Please consult a professional.",
            "ar": f"المشكلة المكتشفة: {class_label}. يُفضل استشارة خبير نباتات."
        })

        return jsonify({
            "prediction": class_label,
            "advice": diagnosis.get(lang, diagnosis["en"])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
