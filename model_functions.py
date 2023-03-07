import numpy as np
import transformers 
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from torch import nn
from utils import args
import streamlit as st


@st.cache_resource
def load_model(base_model_path,tuned_model_path):
    model = AutoModelForQuestionAnswering.from_pretrained(base_model_path)
    model = nn.DataParallel(model)
    device = torch.device("cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(tuned_model_path,map_location="cpu"))
    model = model.module
    return model.eval()

@st.cache_resource
def get_tokenizer(base_model_path):
    return AutoTokenizer.from_pretrained(base_model_path,local_files_only=True)


def get_prediction(model,question,context,tokenizer):
    tokenized_data = tokenizer(
        question,context,
        max_length = args().max_len,
        stride = args().doc_stride,
        truncation = "only_second",
        padding = "max_length",
        return_overflowing_tokens = True,
        return_offsets_mapping = True,
        return_tensors = "pt"
    )
    output = model(tokenized_data['input_ids'], tokenized_data['attention_mask'])
    return output.start_logits, output.end_logits, tokenized_data['input_ids']

def postprocess(tokenizer,start_logits,end_logits,input_ids,n_best=20):

    start_logits = start_logits[0].detach().cpu().numpy()
    start_idxs = np.argsort(start_logits)[-1:-n_best:-1]
    end_logits = end_logits[0].detach().cpu().numpy()
    end_idxs = np.argsort(end_logits)[-1:-n_best:-1]

    score = -np.inf
    start_index = 0
    end_index = 0
    for i in start_idxs:
        for j in end_idxs:
            if i <= j and score < start_logits[i]+end_logits[j]:
                score = start_logits[i]+end_logits[j]
                start_index = i
                end_index = j
                
    result = tokenizer.decode(input_ids[0][start_index: end_index+1])          
    return result, score

