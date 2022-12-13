import sys
sys.path.append('./')

import collections
import random
from typing import Tuple, List
import json
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn
from torch.nn import CosineSimilarity
from transformers import BertModel, BertTokenizer

from dpr.models.biencoder import BiEncoder

logger = logging.getLogger()

from torch.serialization import default_restore_location

checkpoint_dir = "/home/intern2/seungone/ConEV/checkpoints/dpr_biencoder.4"
test_file_dir = "/home/intern2/seungone/ConEV/data/raw/defeasible_snli_test.json"

CheckpointState = collections.namedtuple(
    "CheckpointState",
    [
        "model_dict",
        "optimizer_dict",
        "scheduler_dict",
        "offset",
        "epoch",
        "encoder_params",
    ],
)
def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model

def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info("Reading saved model from %s", model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    logger.info("model_state_dict keys %s", state_dict.keys())
    return CheckpointState(**state_dict)



if __name__ == "__main__":
    naive_score={}
    naive_score['rougeL']=0.0
    naive_score['bertscore']=0.0
    naive_score['sacrebleu']=0.0
    naive_score['meteor']=0.0
    naive_score['cossim']=0.0
    
    cot_score={}
    cot_score['rougeL']=0.0
    cot_score['bertscore']=0.0
    cot_score['sacrebleu']=0.0
    cot_score['meteor']=0.0
    cot_score['cossim']=0.0
    
    question_encoder = BertModel.from_pretrained('bert-large-uncased')
    context_encoder = BertModel.from_pretrained('bert-large-uncased')
    
    states = load_states_from_checkpoint(checkpoint_dir)
    
    model = BiEncoder(question_encoder,context_encoder,True,True)
    model.load_state(states)
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    cos_sim = CosineSimilarity(dim=0,eps=1e-6)
    
    with open(test_file_dir,'r') as f:
        json_object = json.load(f)
        print(len(json_object))
        for data in tqdm(json_object):
            
            p = data['premise']
            h = data['hypothesis']
            if p[-1] != '.':
                p += '.'
            if h[-1] != '.':
                h += '.'
            q = "[CLS] "+p+ " [SEP] "+h
            q_enc = tokenizer(q,return_tensors='pt')
            q_output = model.question_model(q_enc['input_ids'],q_enc['token_type_ids'],q_enc['attention_mask'])
            q_output = q_output['last_hidden_state'].squeeze(0)[0,:]
            
            # choose among naive
            idx = random.choice(range(0,5))
            naive_score['rougeL'] += data['strengthener_prediction']['naive'][idx]['rougeL']
            naive_score['bertscore'] += data['strengthener_prediction']['naive'][idx]['bertscore']
            naive_score['sacrebleu'] += data['strengthener_prediction']['naive'][idx]['sacrebleu']
            naive_score['meteor'] += data['strengthener_prediction']['naive'][idx]['meteor']
            naive_score['cossim'] += data['strengthener_prediction']['naive'][idx]['cossim']
            
            # choose among cot
            max_sim = 0.0
            final_d = {}
            for idx,d in enumerate(data['strengthener_prediction']['cot']):
                exp = d['explanation']
                exp = "[CLS] "+exp
                exp_enc = tokenizer(exp,return_tensors='pt')

                exp_output = model.ctx_model(exp_enc['input_ids'],exp_enc['token_type_ids'],exp_enc['attention_mask'])
                exp_output = exp_output['last_hidden_state'].squeeze(0)[0,:]                           
                
                score = cos_sim(q_output,exp_output)
                if score>max_sim:
                    max_sim = score
                    final_d = d
            cot_score['rougeL'] += final_d['rougeL']
            cot_score['bertscore'] += final_d['bertscore']
            cot_score['sacrebleu'] += final_d['sacrebleu']
            cot_score['meteor'] += final_d['meteor']
            cot_score['cossim'] += final_d['cossim']
    
        for key in naive_score:
            naive_score[key] /= len(json_object)
        for key in cot_score:
            cot_score[key] /= len(json_object)
        print(naive_score)
        print()
        print(cot_score)
                
            
    
    
    
    