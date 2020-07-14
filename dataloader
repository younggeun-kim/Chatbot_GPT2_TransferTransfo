class Custom_Dataset(Dataset):
  def __init__(self, input_ids, cls_label):
    self.input_ids = input_ids
    self.cls_label = cls_label 
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    input_ids = self.input_ids[idx]
    token_type_ids = make_token_type_ids(self.input_ids[idx])
    LM_label = make_LMlabel(self.input_ids[idx])
    cls_label = self.cls_label[idx]

    return input_ids, token_type_ids, LM_label, cls_label
    
chat_dataset = Custom_Dataset(fin_input, fin_label)

Batch_size=16

dataloader = DataLoader(chat_dataset, batch_size=Batch_size, shuffle=True)
