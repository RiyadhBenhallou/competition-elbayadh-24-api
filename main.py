from flask import Flask, request, jsonify
from flask_cors import CORS
from model import * 

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def funct():
    data = request.get_json()
    
    user_attributes = {
        'likes_music': bool(data.get('likes_music', False)),
        'studies_at_night': bool(data.get('studies_at_night', False)),
        'smokes': bool(data.get('smokes', False)),
        'health_issues': bool(data.get('health_issues', False)),
        'likes_reading': bool(data.get('likes_reading', False)),
        'drinks_coffee': bool(data.get('drinks_coffee', False)),
        'exercises_regularly': bool(data.get('exercises_regularly', False)),
        'prefers_group_study': bool(data.get('prefers_group_study', False))
    }
    
    print("Received attributes:", user_attributes)
    
    prediction = predict_matches(user_attributes)
    
    serializable_data = [
        {"name": name, "score": float(score), "id": int(index)} 
        for name, score, index in prediction
    ]
    
    return jsonify({'prediction': serializable_data})

if __name__ == '__main__':
    app.run(debug=True)
