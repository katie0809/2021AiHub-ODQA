# 플라스크 적용
from flask import Flask, request
from flask_api import status
import json

from casualtalker import CasualTalker

app = Flask(__name__)
casualtalker = CasualTalker()

@app.route('/')
def greeting():
    return "This is Tensorflow Python API ! "

@app.route('/casualtalk', methods=['POST'])
def get_casual_response():
    print("In Casual => ", request)
    if request.method == 'POST':
        print("request", request)
        
        params = request.get_json()
        print("request body ", params)
        question = params['userRequest']['utterance']
        try:
            result = casualtalker.return_similar_answer(question)
            resbody = { "version": '2.0',
		 			    "template": {
		 				    "outputs": [
		 					    {
		 						    "simpleText": {
		 							"text": result['answer']
		 						 },
		 					    },
		 				    ],
                         }
            }

            jsonres = json.dumps(resbody, ensure_ascii=False)
            print(jsonres)

            return jsonres, status.HTTP_200_OK, {"Content-Type": "application/json; charset=utf-8", "Access-Control-Allow-Origin": "*"}

        except Exception as e:
            raise Exception('Fail to get casual response', e)
