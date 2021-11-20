# 플라스크 적용
from flask import Flask, request
from flask_api import status
import json

from predictor import QAPrediction

app = Flask(__name__)

configs = {
    'model_name_or_path': '/home/ubuntu/2021AiHub-ODQA/models/korquad_2/0-0-ckpt',
    'max_seq_len': 4096,
    'doc_stride': 512
}
print('hello')
predictor = QAPrediction(configs)

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
        # print("result", result['answer'])
        # answer = predictor.predict([context],[question])
        # result = json.dumps(answer[0]["answer"][0]["answer"],ensure_ascii=False)