# 플라스크 적용
from flask import Flask, request
from flask_api import status
import json

from casualtalker import CasualTalker
from predictor import QAPrediction

app = Flask(__name__)

pred_configs = {
    'model_name_or_path': '/home/ubuntu/2021AiHub-ODQA/models/korquad_2/0-0-ckpt',
    'max_seq_len': 4096,
    'doc_stride': 512
}
predictor = QAPrediction(pred_configs)
casualtalker = CasualTalker()

@app.route('/')
def greeting():
    return "This is Tensorflow Python API ! "
    
@app.route('/predict', methods=['POST'])
def get_predict():
    if request.method == 'POST':
        # print("request", request)
        question = request.json["question"]
        # print("question", question)
        context = request.json["context"]
        # print("context", context)
        try:
            result = predictor.predict(context, question)
            jsonres = json.dumps(result, ensure_ascii=False)

            return jsonres, status.HTTP_200_OK, {"Content-Type": "application/json; charset=utf-8", "Access-Control-Allow-Origin": "*"}

        except Exception as e:
            raise Exception('Fail to predict', e)

@app.route('/casualtalk', methods=['POST'])
def get_casual_response():
    if request.method == 'POST':
        # print("request", request)
        question = request.json["question"]
        # print("question", question)
        
        try:
            result = casualtalker.return_similar_answer(question)
            jsonres = json.dumps(result, ensure_ascii=False)

            return jsonres, status.HTTP_200_OK, {"Content-Type": "application/json; charset=utf-8", "Access-Control-Allow-Origin": "*"}

        except Exception as e:
            raise Exception('Fail to get casual response', e)