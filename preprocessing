import json

# Download and load JSON dataset
with open('personachat_self_original.json', "r", encoding="utf-8") as f:
    dataset = json.loads(f.read())

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")

special_tokens = ['<SP1>', '<SP2>', '<PAD>']
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
special_ids = tokenizer.convert_tokens_to_ids([
        '<SP1>', '<SP2>', tokenizer.bos_token,
        tokenizer.eos_token,  '<PAD>'])
GPT2model.resize_token_embeddings(len(tokenizer)) #tokenizer model에 적용        

    
personality=[]
utterances=[]
for i in range(len(dataset['train'])):
  personality.append(dataset['train'][i]['personality'])
  utterances.append(dataset['train'][i]['utterances'])
persona=[]
history=[]
spokerA=[]
spokerB_real=[]
spokerB_distract1=[]
spokerB_distract2=[]
#spokerB_distract3=[]
for i in range(len(personality)-1):
  for j in range(len(utterances[i])):
    candidate = utterances[i][j]['candidates']
    history1 = utterances[i][j]['history']
    persona.append(personality[i])
    history.append(history1)
    spokerA.append(history1[-1])
    spokerB_real.append(candidate[-1])
    spokerB_distract1.append(candidate[-2])
    spokerB_distract2.append(candidate[-3])  

#persona+ <SP1>+ spokerA+ <SP2>+ spokerB
def make_input(persona, spokerA, spokerB):
  input=[]
  for i in range(len(persona)):
    ids=[]
    for j in range(len(persona[i])):
      ids.append(persona[i][j])
    spA='<SP1> '+spokerA[i]  
    spB='<SP2> '+spokerB[i]  
    ids.append(spA)
    ids.append(spB)  
    input.append(ids)

  return input  
  
#persona+ <SP1>+ history + <SP2>+ spokerB
def make_input_H(persona, history, spokerB):
  input=[]
  for i in range(len(persona)):
    ids=[]
    for j in range(len(persona[i])):
      ids.append(persona[i][j]) 
    for k in range(len(history[i])):
      if k ==0:
        ids.append('<SP1> '+history[i][k])
      else: ids.append(history[i][k])
    spB='<SP2> '+spokerB[i]  
    ids.append(spB)  
    input.append(ids)

  return input  
  
real_H_data = make_input_H(persona, history, spokerB_real)
fake_H_data1 = make_input_H(persona, history, spokerB_distract1)

#<bos>+[persona]+<SP2>+[utr]+<SP1>+[utr]+<EOS>+<PAD>
def make_input_ids(input, max_len=96):
  input_ids = []
  max_len=max_len
  for i in range(len(input)):
    ids = []
    for j in range(len(input[i])):
      ids += tokenizer.encode(input[i][j])
    length = len(ids)+2  
    ids = tokenizer.encode(tokenizer.bos_token)+ids+tokenizer.encode(tokenizer.eos_token)
    +tokenizer.encode('<PAD>')*(max_len-length) 
    ids = ids[:max_len] 
    input_ids.append(ids)
  return torch.tensor(input_ids)  
  
input_ids_R = make_input_ids(real_H_data)
input_ids_F1 = make_input_ids(fake_H_data1)

def make_token_type_ids(input_str):
  x=0    
  #input_str=input_str.view(-1)
  tok=[]
  for i in range(len(input_str)):
    if input_str[i]==50257:
      x=1
    if input_str[i]==50258:
      x=2  
    #if input_str[i]==50259:
    #  x=3  
    tok.append(x) 
             
  return torch.tensor(tok)  
  
  
def make_LMlabel(input_ids):
    input=input_ids.squeeze()
    label=torch.zeros(size=input.size(), dtype=torch.long)
    for i in range(len(input)-1):
      label[i]=input[i+1]
    if input[len(input)-1] == 50259:
      label[len(input)-1]=50259
    else: label[len(input)-1]=50256

    return label.unsqueeze(0)    
    
cls_label_R = torch.ones(size=(input_ids_R.size(0),1), dtype=torch.long)
cls_label_F1 = torch.zeros(size=(input_ids_F1.size(0),1), dtype=torch.long)

def concat_input(input_ids1, input_ids2):
  return torch.cat((input_ids1, input_ids2), dim=0)
  
fin_input = concat_input(input_ids_R, input_ids_F1)
fin_label = concat_input(cls_label_R , cls_label_F1) 
