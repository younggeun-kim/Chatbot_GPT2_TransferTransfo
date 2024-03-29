#postprocessing 부분은 수정해야 할 부분이 많은 것 같습니다 참고용도로만 봐주세요

def make_input_chat(input): #interact에 사용될 chat을 알맞는 input의 형태로 변환시켜준다.
  input_ids = []
  ids = []
  for j in range(len(input)):
      ids += tokenizer.encode(input[j]) 
  ids = tokenizer.encode(tokenizer.bos_token)+ids+[50258]
  input_ids.append(ids)
  return torch.tensor(input_ids)  
  
def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)  
    
def find_num(aa,num):
  s=0
  for i in range(len(aa.view(-1))):
    aa=aa.view(-1)
    #print(aa.cpu().detach().numpy().tolist()[i])
    
    if (aa.cpu().detach().numpy().tolist()[i]) == (num):
    
      s=s+1  
       
  return s    
  
def generate_some_text(input_str,token_type_ids, model): #50259는 [PAD], 50257은 [SP1](spoker1), 50258은 [SP2](spoker2)를 의미한다.
    cur_ids = input_str
    length = (cur_ids.size(-1))
    model.eval()
    with torch.no_grad():
      
      while find_num(cur_ids, 50259)<3:

            outputs = model(cur_ids, token_type_ids)
            LM_logits, CLS_logits = outputs
            softmax_logits = torch.softmax(LM_logits[0,-1], dim=0) #Take the first(only one) batch and the last predicted embedding
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=5) #Randomly(from the given probability distribution) choose the next word from the top n words

            if next_token_id == 50259:
              token_type_ids= torch.cat([token_type_ids, torch.ones((1,1)).long().to(device)*2], dim=1)
            else:
              if find_num(cur_ids, 50258)>find_num(cur_ids, 50257): 
                if next_token_id == 50258:
                  next_token_id = 50257
                  token_type_ids= torch.cat([token_type_ids, torch.ones((1,1)).long().to(device)*1], dim=1)
                else:
                  token_type_ids= torch.cat([token_type_ids, torch.ones((1,1)).long().to(device)*2], dim=1)

              elif  find_num(cur_ids, 50258)<=find_num(cur_ids, 50257):  
                if next_token_id == 50257:                  
                  next_token_id = 50258                 
                  token_type_ids= torch.cat([token_type_ids, torch.ones((1,1)).long().to(device)*2], dim=1)  
                else: token_type_ids= torch.cat([token_type_ids, torch.ones(1,1).long().to(device)*1], dim=1)    

            cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word 
            
            if cur_ids.size(-1)-length>30:
              break
            
   

      output_list = list(cur_ids.squeeze().to('cpu').numpy())
      output_text = tokenizer.decode(output_list)
  
      return output_text, cur_ids, token_type_ids, length  
      
def cls_some_text(cur_ids, token_type_ids, model, list_cls):

    model.eval()
    with torch.no_grad():
      _, cls_outputs = model(cur_ids, token_type_ids)
      cls = nn.Sigmoid()(cls_outputs)
      list_cls.append(cls)

    return list_cls
    
def chat(personality, history, tokenizer, model=chatbot, num=3): 
  
  #input_str 토큰화 하기
  speechs = []
  list_cls = []
  input=[]
  for j in range(len(personality)):
      input.append(personality[j])   
  for k in range(len(history)):
      if k ==0:
        input.append('<SP1> '+history[k])
      else: input.append(history[k])

  input_str = make_input_chat(input)
  input_str2=input_str.view(-1)
  token_type_ids = make_token_type_ids(input_str2)
  input_str = input_str.to(device)
  token_type_ids = token_type_ids.view(1,-1).to(device)

  for i in range(num):
    o, c, t, l = generate_some_text(input_str,token_type_ids, chatbot)
    speechs.append(o)
    list_cls = cls_some_text(c, t, chatbot, list_cls)
  idx = np.argmax(list_cls)
  speech = speechs[idx]

  return speech, l    
  
def interact(personality, model):
  history = []
  chatbo=chatbot_model() #model
  chatbo=chatbo.to(device)
  model = torch.load(PATH+'model:2')
  while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        history.append(raw_text)
        with torch.no_grad():
            out_ids, l = chat(personality, history, tokenizer, model)
        out = tokenizer.encode(out_ids)
        spoker = out[l:]
        out_text = tokenizer.decode(out, skip_special_tokens=True)
        spoker_text = tokenizer.decode(spoker, skip_special_tokens=True)
        history.append(spoker_text)

        #print(out_text)
        print(spoker_text)
        
personal = ['i like to remodel homes .',
 'i like to go hunting .',
 'i like to shoot a bow .',
 'my favorite holiday is halloween .']
 
interact(personal, chatbot) #chatbot은 모델이다
