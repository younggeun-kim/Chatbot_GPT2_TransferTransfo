device='cuda'
lr = 2e-5
PATH = '/content/gdrive/My Drive/'
chatbot=chatbot_model() #model
chatbot=chatbot.to(device)


def train_one_epoch(data_loader, model, optimizer, device):
  
  model.train()
  tk0 = tqdm(data_loader, total=len(data_loader))
  total_loss = 0.0
  
  for bi, d in enumerate(tk0):
      input_ids, token_type_ids, LM_label, CLS_label = d
      input_ids = input_ids.to(device, dtype=torch.long)
      token_type_ids = token_type_ids.to(device, dtype=torch.long)
      LM_label = LM_label.to(device, dtype=torch.long)
      CLS_label = CLS_label.to(device, dtype=torch.float)

      model.zero_grad()
      LM_logits, CLS_logits = model(input_ids, token_type_ids)
      LM_logits = LM_logits.view(-1, 50260)
      LM_label = LM_label.view(-1)
      loss_LM = nn.CrossEntropyLoss()(LM_logits, LM_label)
      loss_CLS = nn.BCEWithLogitsLoss()(CLS_logits, CLS_label)*2
      loss = loss_LM + loss_CLS
      total_loss += loss.item()
      if bi % 100 ==0:
          print(f"loss:{loss}")
      if bi % 500 ==0:
          print(f"LM_loss:{loss_LM}") 
          print(f"CLS_loss:{loss_CLS}")  
      loss_LM.backward(retain_graph=True)
      loss_CLS.backward()
      optimizer.step()
      optimizer.zero_grad()

  avg_train_loss = total_loss / len(data_loader) 
  print(" Average training loss: {0:.2f}".format(avg_train_loss))  
  
  
def fit(train_dataloader, EPOCHS=1):
  optimizer = torch.optim.AdamW(chatbot.parameters(),lr=lr) #optimizer

  for i in range(EPOCHS):
    print(f"EPOCHS:{i+1}")
    print('TRAIN')
    train_one_epoch(train_dataloader, chatbot, optimizer, device)    
    torch.save(chatbot ,PATH+f'model:{i+1}')
    
 fit(dataloader)   
