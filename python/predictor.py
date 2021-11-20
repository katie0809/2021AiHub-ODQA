from tqdm import tqdm

import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig
from transformers import BigBirdTokenizer, BigBirdForQuestionAnswering
import torch
import numpy as np

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
        softmax = torch.nn.Softmax(dim=0)
        is_answer_found = False
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
                    input_ids = encodings["input_ids"].cuda()
                    token_type_ids = encodings["token_type_ids"].cuda()
                    attention_mask = encodings["attention_mask"].cuda()
                    
                    outputs = self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)

                    start_score, end_score = float(torch.max(softmax(outputs[0][0]))), float(torch.max(softmax(outputs[1][0])))
                    start_logits, end_logits = outputs[0].to(self.device), outputs[1].to(self.device)
                    token_start_index, token_end_index = start_logits.argmax(dim=-1), end_logits.argmax(dim=-1)
                    pred = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(encodings["input_ids"][0][token_start_index: token_end_index + 1]))
                    pred = pred.replace("#","")
                    if pred != '[CLS]':
                        is_answer_found = True
                        answers.append({
                            'answer_start': int(token_start_index.cpu().numpy()[0]), 
                            'answer_end': int(token_end_index.cpu().numpy()[0]), 
                            'answer': pred,
                            'answer_score': [start_score, end_score]
                        })

                result.append({
                    'context': context,
                    'question': question,
                    'answer': answers
                })
                print('answers', answers)

        if is_answer_found:
            return result  
        else:
            return []

    def _softmax(self, n):
        exp_n = np.exp(n)
        sum_exp_n = np.sum(exp_n)
        return exp_n / sum_exp_n

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