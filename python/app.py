# 플라스크 적용
from flask import Flask, request
from flask_api import status
import json

from predictor import QAPrediction
from classifier import QAClassification

app = Flask(__name__)

pred_configs = {
    'model_name_or_path': '/home/ubuntu/2021AiHub-ODQA/models/korquad_2/0-0-ckpt',
    'max_seq_len': 4096,
    'doc_stride': 512
}
intent_configs = {
    'model_name_or_path': '/home/ubuntu/2021AiHub-ODQA/models/korquad_2/0-0-ckpt',
    'max_seq_len': 4096,
    'doc_stride': 512
}
predictor = QAPrediction(pred_configs)

@app.route('/')
def greeting():
    return "This is Tensorflow Python API ! "
    
@app.route('/predict', methods=['POST'])
def get_predict():
    if request.method == 'POST':
        print("request", request)
        question = request.json["question"]
        print("question", question)
        context = request.json["context"]
        # print("context", context)
        try:
            result = predictor.predict(context, question)
            jsonres = json.dumps(result, ensure_ascii=False)

            return jsonres, status.HTTP_200_OK, {"Content-Type": "application/json; charset=utf-8", "Access-Control-Allow-Origin": "*"}

        except Exception as e:
            raise Exception('Fail to predict', e)

@app.route('/intent', methods=['POST'])
def get_intent():
    classifier = QAPrediction(intent_configs)
    if request.method == 'POST':
        print("request", request)
        question = request.json["question"]
        try:
            result = classifier.get_intent(question)
            jsonres = json.dumps(result, ensure_ascii=False)

            return jsonres, status.HTTP_200_OK, {"Content-Type": "application/json; charset=utf-8", "Access-Control-Allow-Origin": "*"}

        except Exception as e:
            raise Exception('Fail to predict', e)