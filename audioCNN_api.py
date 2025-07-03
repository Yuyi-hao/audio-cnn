from ml_code.main import predict_audio_from_model
from flask import Flask, jsonify, request
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])


@app.route("/predict/", methods=['POST', 'GET'])
def predict_sound():
    data = request.data
    data = data.decode('utf-8')
    data = json.loads(data)
    audio_data = data['audio_data']
    result = predict_audio_from_model(audio_data)
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True)