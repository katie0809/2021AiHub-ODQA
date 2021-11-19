from tqdm import tqdm

import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig
from transformers import BigBirdTokenizer, BigBirdForQuestionAnswering
import torch

import gc

class QAPrediction():
    def __init__(self, config):
        self.config = config

        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.config['model_name_or_path'],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name_or_path'], 
            use_fast=False
        )
        # torch cude 확인
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        print("The model will be running on", self.device, "device\n") 
        self.model.to(self.device)  

    def predict(self, contexts, questions):
        result = []
        
        with torch.no_grad():
            for context, question in tqdm(zip(contexts, questions)):
                answers = []
                max_seq_len = self.config['max_seq_len']
                chunks = len(context) // max_seq_len
                chunks += 1 if len(context) % max_seq_len > 0 else 0

                for i in range(chunks):
                    stpos = i*max_seq_len
                    endpos = min(len(context), (i+1)*max_seq_len-self.config['doc_stride'])

                    encodings = self.tokenizer(question, 
                                        context[stpos:endpos], 
                                        return_tensors="pt")
                    
                    # input 을 cuda로 저장
                    print("type ",type(encodings))
                    print(encodings)

                    # input_ids = encodings["input_ids"].to(self.device).cpu()
                    # token_type_ids = encodings["token_type_ids"].to(self.device).cpu()
                    # attention_mask = encodings["attention_mask"].to(self.device).cpu()
                    input_ids = encodings["input_ids"].cuda()
                    token_type_ids = encodings["token_type_ids"].cuda()
                    attention_mask = encodings["attention_mask"].cuda()
                    # input_ids = encodings["input_ids"]
                    # token_type_ids = encodings["token_type_ids"]
                    # attention_mask = encodings["attention_mask"]
                    
                    outputs = self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)

                    # outputs = self.model(**encodings)
                  

                    start_logits, end_logits = outputs[0].to(self.device), outputs[1].to(self.device)
                    token_start_index, token_end_index = start_logits.argmax(dim=-1), end_logits.argmax(dim=-1)
                    pred = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(encodings["input_ids"][0][token_start_index: token_end_index + 1]))
                    pred = pred.replace("#","")
                    if pred != '[CLS]':
                        answers.append({
                            'answer_start': token_start_index.cpu().numpy()[0], 
                            'answer_end': token_end_index.cpu().numpy()[0], 
                            'answer': pred
                        })

                result.append({
                    'context': context,
                    'question': question,
                    'answer': answers
                })

        return result

    def compute_exact(self, a_gold, a_pred):
        return int(normalize_answer(a_gold) == normalize_answer(a_pred))

    def compute_f1(self, a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def normalize_answer(self, sentence):
        ret = re.sub('[-=+,#/\:^$@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', sentence) # 특수문자 제거
        ret = re.sub(r'[\n\r\t]+', ' ', sentence) # 개행문자 제거
        ret = ret.strip() # 양쪽공백 제거
        return ret

    def get_tokens(self, s):
        if not s: return []
        return normalize_answer(s).split()


###===============================
# SAMPLE EXECUTION CODE
###===============================
# configs = {
#     'model_name_or_path': '/home/ubuntu/kylee/output/qa/korquad_2/0-0-ckpt',
#     'max_seq_len': 1024,
#     'doc_stride': 128
# }
configs = {
    'model_name_or_path': '/home/ubuntu/2021AiHub-ODQA/models/korquad_2/0-0-ckpt',
    'max_seq_len': 4096,
    'doc_stride': 512
}
# configs = {
#     'model_name_or_path': '/home/ubuntu/2021AiHub-ODQA/models/korquad_2/0-0-ckpt',
#     'max_seq_len': 1024,
#     'doc_stride': 128
# }
predictor = QAPrediction(configs)
# results = predictor.predict(   
# [
#     '주차장에서 한 여성이 피투성이가 된 채 발견된다. 피해자는 니콜 매닝. 놀랍게는 니콜은 임산부였고 누군가에게 폭행당한 후 제왕절개술로 아이를 뺏긴 채 쓰러져 있었다. 니콜에게 원한이 있는 사람을 중심으로 수사하던 성범죄 전담반은 니콜이 간호사로 일한 메타딘 클리닉의 환자였던 카일 노바첵과 니콜에게 부정 행위를 들켜 실직된 전직 간호사 에린 두 사람을 의심한다. 하지만 카일은 완벽한 알리바이를 제공하고 뭔가를 숨기는 듯한 에린 역시 더 이상 수사에 도움이 되지는 않는다. ',
#     "The Ever Given is 400m-long (1,312ft) and weighs 200,000 tonnes, with a maximum capacity of 20,000 containers. It is currently carrying 18,300 containers."
# ],
# [
#     '피해자의 이름은 무엇인가?',
#     "How heavy is Ever Given?"
# ])
# print(results)


# 플라스크 적용
from flask import Flask, request
from flask_api import status
import json

app = Flask(__name__)

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
        print("context", context)
        answer = predictor.predict(context,question)
        # answer = predictor.predict([context],[question])
        print("answer :", type(answer), answer)
        print(type(answer[0]))
        # result = json.dumps(answer[0]["answer"][0]["answer"],ensure_ascii=False)
        answer_list = []
        for asw in answer:
            if asw["answer"]:
                answer_list.append(asw["answer"][0]["answer"])
        
        result = json.dumps(answer_list,ensure_ascii=False)

    return result, status.HTTP_200_OK, {"Content-Type": "application/json; charset=utf-8", "Access-Control-Allow-Origin": "*"}
