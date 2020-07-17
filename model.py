import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel, GPT2LMHeadModel, GPT2Model

GPT2model = GPT2Model.from_pretrained("microsoft/DialoGPT-medium") #작성자는 DialoGPT를 활용하였다.

class chatbot_model(nn.Module):
  def __init__(self):
    super(chatbot_model, self).__init__()
    self.model = GPT2model
    self.drop = nn.Dropout(0.3)
    self.line = nn.Linear(1024*2,1)
    self.LM = nn.Linear(1024,50260)
    self.m = nn.Sigmoid()

  def forward(self, input, token):
    out = self.model(input, token_type_ids= token)
    x = out[0]
    x = self.drop(x)
    #print(x.size())
    apool=torch.mean(x,1)
    mpool,_=torch.max(x,1)
    #print(apool.size())
    #print(mpool.size())
    catcat=torch.cat((apool,mpool),1)
    #print(catcat.size())
    catcat = self.line(catcat)
    x = self.LM(x)
    return x, catcat
    
chatbot=chatbot_model() #model
chatbot=chatbot.to(device)
