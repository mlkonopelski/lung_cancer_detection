from flask import Flask, request, jsonify
import torch
import numpy as np
import os
import sys
import json
from typing import Dict


flask_app = Flask(__name__)
cls_model = ...

def run_interferance(tensor: torch.Tensor) -> Dict:
    with torch.no_grad():
        prediction = cls_model(tensor)
    response = {'probs': prediction.tolist()[1]}
    return response

@flask_app.route('/predict', methods=['POST'])
def predict():
    meta = json.load(request.files['meta'])
    blob = request.files['blob'].read()
    in_tensor = torch.from_numpy(np.frombuffer(blob, dtype=np.float32))
    in_tensor = in_tensor.view(meta['shape'])
    out = run_interferance(in_tensor)
    return jsonify(out)


if __name__ == '__main__':
    
    flask_app.run(host='0.0.0.0', port=8000)
