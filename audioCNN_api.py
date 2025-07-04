from ml_code.main import predict_audio_from_model
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
from dotenv import load_dotenv
import traceback

load_dotenv()

app = Flask(__name__)
CORS(app, origins=[os.getenv("FRONTEND_URL")])


@app.route("/predict/", methods=['POST'])
def predict_sound():
    try:
        data = request.data
        data = data.decode('utf-8')
        data = json.loads(data)
        audio_data = data['audio_data']
        if not audio_data:
            return jsonify({"error": "Missing 'audio_data' in request"}), 400
        
        result = predict_audio_from_model(audio_data)
        return jsonify(result)
    
    except Exception as e:
        traceback.print_exc()  # logs error in server logs
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500
        

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.getenv("DEBUG")=="True")