from transformers import BertTokenizer
import json
from tqdm import tqdm
strengthener = "/home/intern2/seungone/ConEV/data/dpr_style/defeasible_strengthener_snli_train.json"
weakener = "/home/intern2/seungone/ConEV/data/dpr_style/defeasible_weakener_snli_train.json"

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

def find_length(x):
    t = tokenizer(x)['input_ids']
    return len(t)

q_len = []
exp_len = []
with open(strengthener,'r') as f:
    json_obj = json.load(f)
    for data in tqdm(json_obj):
        q_len.append(find_length(data['question']))
        exp_len.append(find_length(data['positive_ctxs'][0]['text']))
        for d in data['hard_negative_ctxs']:
            exp_len.append(find_length(d['text']))

print(sum(q_len)/len(q_len))
print(max(q_len))
print(sum(exp_len)/len(exp_len))
print(max(exp_len))
ans=0
for q in q_len:
    if q>64:
        ans+=1
print(ans/len(q_len))
ans=0
for exp in exp_len:
    if exp>64:
        ans+=1
print(ans/len(exp_len))

with open(weakener,'r') as f:
    json_obj = json.load(f)
    for data in tqdm(json_obj):
        q_len.append(find_length(data['question']))
        exp_len.append(find_length(data['positive_ctxs'][0]['text']))
        for d in data['hard_negative_ctxs']:
            exp_len.append(find_length(d['text']))
            
print(sum(q_len)/len(q_len))
print(max(q_len))
print(sum(exp_len)/len(exp_len))
print(max(exp_len))
ans=0
for q in q_len:
    if q>64:
        ans+=1
print(ans/len(q_len))
ans=0
for exp in exp_len:
    if exp>64:
        ans+=1
print(ans/len(exp_len))