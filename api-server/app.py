import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import google.generativeai as genai
from google.ai.generativelanguage import Content, Part, Blob
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables for API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# If .env not found, fallback (local testing only)
if not API_KEY:
    API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # <-- Replace this if you don't have .env

# Configure Gemini
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    raise Exception(f"Failed to configure Gemini API: {e}")

# Initialize output parser and model
parser = StrOutputParser()
llm = genai.GenerativeModel("gemini-2.5-pro")

# Allow image file uploads and restrict file types
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# Store analysis results and form data in memory (for simplicity)
data_store = {"analysis_result": None, "form_data": None}


# Helper function to validate file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


# POST route for analyzing data
@app.route("/analyze", methods=["POST"])
def analyze_data():
    try:
        # Validate file input
        if "image" not in request.files:
            return jsonify({"error": "No image file part"}), 400

        image_file = request.files["image"]

        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            image_file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image_file.save(image_file_path)

            # Extract form data
            form_data = request.form.to_dict()
            vendor_name = form_data.get("vendorName", "")
            business_name = form_data.get("businessName", "")
            business_title = form_data.get("businessTitle", "")
            location = form_data.get("location", "")
            welfare_scheme = form_data.get("welfareScheme", "")
            family_members = form_data.get("familyMembers", "")
            shop_location = form_data.get("shopLocation", "")
            government_ids = form_data.get("governmentIds", "")

            # Gemini API prompt
            generic_template = f"""
You are a knowledgeable AI assistant. Analyze the uploaded image of a vendor shop and provide a detailed report using the following format:
    business_activity_observations: "Number of persons visible in the image",
    asset_Ownership: "Comma-separated list of assets visible",
    seasonal_and_env_impact: "High Seasonal Variation, Moderate Seasonal Variation, Minimal Seasonal Variation, or Weather Sensitive",
    business_location: "The shop is located in {location}. Classify as Prime Location, Moderate Traffic, Low Traffic Area, Home-Based, or Remote Area",
    local_market_standing: "The shop name is {business_name}, located at {location}. Rate as Highly Popular, Moderately Popular, Average, Low Footfall, New Business, or Business Not Found",
    customer_review: "Range from 1 to 10 based on Google reviews, or 'No Reviews Found'"
No extra explanation of the image is needed; only the above information is required. Do not return JSON format.
            """

            # Prepare Gemini content parts
            file_type = image_file.mimetype
            image_file.seek(0)
            image_data = image_file.read()
            content_parts = [
                Part(text=generic_template),
                Part(inline_data=Blob(mime_type=file_type, data=image_data))
            ]

            # Generate content from Gemini
            response = llm.generate_content(Content(parts=content_parts))

            # Safely extract model output text
            if (
                response
                and response.candidates
                and response.candidates[0].content.parts
                and hasattr(response.candidates[0].content.parts[0], "text")
            ):
                parsed_response = parser.invoke(response.candidates[0].content.parts[0].text)
            else:
                parsed_response = "No valid response received from Gemini."

            # Store data in memory
            data_store["analysis_result"] = parsed_response
            data_store["form_data"] = {
                "vendor_name": vendor_name,
                "business_name": business_name,
                "business_title": business_title,
                "location": location,
                "welfare_scheme": welfare_scheme,
                "family_members": family_members,
                "shop_location": shop_location,
                "government_ids": government_ids,
            }

            return jsonify({"message": "Image analyzed successfully"}), 200

        else:
            return jsonify({"error": "Invalid image file type"}), 400

    except Exception as e:
        print("❌ ERROR TRACEBACK:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# GET route to retrieve analysis data
@app.route("/get-analysis", methods=["GET"])
def get_analysis():
    if data_store["analysis_result"] and data_store["form_data"]:
        return (
            jsonify(
                {
                    "analysis_result": data_store.get("analysis_result", {}),
                    "form_data": data_store["form_data"],
                }
            ),
            200,
        )
    else:
        return jsonify({"error": "No analysis data found. Please upload and analyze an image first."}), 404


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
