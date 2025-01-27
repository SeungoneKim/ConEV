import json
from tqdm import tqdm
import copy

positive_file = "./raw/defeasible_snli_positive.json"
negative_file = "./raw/defeasible_snli_negative.json"

strengthener_explanations = []
weakener_explanations = []

with open(positive_file,'r') as pf:
    with open(negative_file,'r') as nf:
        positive_data = json.load(pf)
        negative_data = json.load(nf)
        print(len(positive_data))
        print(len(negative_data))
        positive_data = positive_data[:len(negative_data)]
        
        for idx,(pd,nd) in enumerate(zip(positive_data, negative_data)):
            if ('strengthener_prediction' in pd) and ('weakener_prediction' in pd) and ('strengthener_prediction' in nd) and ('weakener_prediction' in nd):
                tmp = {}
                tmp['dataset'] = "defeasible_snli"
                if pd['premise'][-1] != '.':
                    premise = pd['premise'] + '.'
                else:
                    premise = pd['premise']
                if pd['hypothesis'][-1] != '.':
                    hypothesis = pd['hypothesis']+ '.'
                else:
                    hypothesis = pd['hypothesis']
                tmp['question'] = premise + ' [SEP] ' + hypothesis
                
                tmp_strengthener = tmp
                tmp_weakener = copy.deepcopy(tmp)
                tmp_strengthener['answers'] = pd['strengthener_answers']
                tmp_weakener['answers'] = pd['weakener_answers']
                
                # Making positive, hard negatives for strengthener
                tmp_strengthener['positive_ctxs'] = []
                tmp_strengthener['positive_ctxs'].append(
                    {
                        "title":pd['strengthener_prediction']['cot'][0]['pred'],
                        "text":pd['strengthener_prediction']['cot'][0]['explanation'] + " [SEP] "+pd['strengthener_prediction']['cot'][0]['pred'],
                        "score":pd['strengthener_prediction']['cot'][0]['cossim']*1000,
                        "title_score":1,
                        "passage_id":idx
                    }
                )
                #tmp_strengthener['negative_ctxs'] = []
                tmp_strengthener['hard_negative_ctxs'] = []
                tmp_strengthener['hard_negative_ctxs'].append(
                    {
                        "title":pd['weakener_prediction']['cot'][0]['pred'],
                        "text":pd['weakener_prediction']['cot'][0]['explanation'] + " [SEP] "+pd['weakener_prediction']['cot'][0]['pred'],
                        "score":1000-pd['weakener_prediction']['cot'][0]['cossim']*1000,
                        "title_score":0,
                        "passage_id":len(positive_file)+idx
                    }
                )
                tmp_hard_negative_pred = ""
                tmp_hard_negative_explanation = ""
                tmp_score = 1.0
                for option in nd['strengthener_prediction']['cot']:
                    if option['cossim']<tmp_score:
                        tmp_hard_negative_pred = option['pred']
                        tmp_hard_negative_explanation = option['explanation']
                        tmp_score = option['cossim']
                tmp_strengthener['hard_negative_ctxs'].append(
                    {
                        "title":tmp_hard_negative_pred,
                        "text":tmp_hard_negative_explanation + " [SEP] "+tmp_hard_negative_pred,
                        "score":tmp_score*1000,
                        "title_score":0,
                        "passage_id":len(positive_file)*2+idx
                    }
                )
                tmp_hard_negative_pred = ""
                tmp_hard_negative_explanation = ""
                tmp_score = 0.0
                for option in nd['weakener_prediction']['cot']:
                    if option['cossim']>tmp_score:
                        tmp_hard_negative_pred = option['pred']
                        tmp_hard_negative_explanation = option['explanation']
                        tmp_score = option['cossim']
                tmp_strengthener['hard_negative_ctxs'].append(
                    {
                        "title":tmp_hard_negative_pred,
                        "text":tmp_hard_negative_explanation + " [SEP] "+tmp_hard_negative_pred,
                        "score":1000-tmp_score*1000,
                        "title_score":0,
                        "passage_id":len(positive_file)*3+idx
                    }
                )
                
                # Making positive, hard negatives for weakener
                tmp_weakener['positive_ctxs'] = []
                tmp_weakener['positive_ctxs'].append(
                    {
                        "title":pd['weakener_prediction']['cot'][0]['pred'],
                        "text":pd['weakener_prediction']['cot'][0]['explanation'] + " [SEP] "+pd['weakener_prediction']['cot'][0]['pred'],
                        "score":1000-pd['weakener_prediction']['cot'][0]['cossim']*1000,
                        "title_score":1,
                        "passage_id":len(positive_file)+idx
                    }
                )
                #tmp_weakener['negative_ctxs'] = []
                tmp_weakener['hard_negative_ctxs'] = []
                tmp_weakener['hard_negative_ctxs'].append(
                    {
                        "title":pd['strengthener_prediction']['cot'][0]['pred'],
                        "text":pd['strengthener_prediction']['cot'][0]['explanation'] + " [SEP] "+pd['strengthener_prediction']['cot'][0]['pred'],
                        "score":pd['strengthener_prediction']['cot'][0]['cossim']*1000,
                        "title_score":0,
                        "passage_id":idx
                    }
                )
                tmp_hard_negative_pred = ""
                tmp_hard_negative_explanation = ""
                tmp_score = 1.0
                for option in nd['weakener_prediction']['cot']:
                    if option['cossim']<tmp_score:
                        tmp_hard_negative_pred = option['pred']
                        tmp_hard_negative_explanation = option['explanation']
                        tmp_score = option['cossim']
                tmp_weakener['hard_negative_ctxs'].append(
                    {
                        "title":tmp_hard_negative_pred,
                        "text":tmp_hard_negative_explanation + " [SEP] "+tmp_hard_negative_pred,
                        "score":tmp_score*1000,
                        "title_score":0,
                        "passage_id":len(positive_file)*2+idx
                    }
                )
                tmp_hard_negative_pred = ""
                tmp_hard_negative_explanation = ""
                tmp_score = 0.0
                for option in nd['strengthener_prediction']['cot']:
                    if option['cossim']>tmp_score:
                        tmp_hard_negative_pred = option['pred']
                        tmp_hard_negative_explanation = option['explanation']
                        tmp_score = option['cossim']
                tmp_weakener['hard_negative_ctxs'].append(
                    {
                        "title":tmp_hard_negative_pred,
                        "text":tmp_hard_negative_explanation + " [SEP] "+tmp_hard_negative_pred,
                        "score":1000-tmp_score*1000,
                        "title_score":0,
                        "passage_id":len(positive_file)*3+idx
                    }
                )
                
                strengthener_explanations.append(tmp_strengthener)
                weakener_explanations.append(tmp_weakener)

print(len(strengthener_explanations))
print(len(weakener_explanations))

train_strengthener_explanations = strengthener_explanations[:6600]
val_strengthener_explanations = strengthener_explanations[6600:]
train_weakener_explanations = weakener_explanations[:6600]
val_weakener_explanations = weakener_explanations[6600:]

with open("./dpr_style/defeasible_strengthener_snli_train.json","w") as f:
    json.dump(train_strengthener_explanations,f)
with open("./dpr_style/defeasible_strengthener_snli_val.json","w") as f:
    json.dump(val_strengthener_explanations,f)
with open("./dpr_style/defeasible_weakener_snli_train.json","w") as f:
    json.dump(train_weakener_explanations,f)
with open("./dpr_style/defeasible_weakener_snli_val.json","w") as f:
    json.dump(val_weakener_explanations,f)