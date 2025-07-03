import torch
from torch.utils.data import Dataset # 데이터로더
from transformers import ElectraTokenizer
import sys
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")
sys.path.append("..")
import options

class ChatbotTextClassificationDataset(Dataset):
  """Wellness Text Classification Dataset"""
  def __init__(self,
               file_path = "../data/hubok_text_classification_train.txt",
               num_label = options.get_label_num(),
               device = 'cpu',
               max_seq_len = 512, # KoBERT max_length
               tokenizer = None
               ):
    self.file_path = file_path
    self.device = device
    self.data =[]

    self.tokenizer = tokenizer if tokenizer is not None else ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")

    file = open(self.file_path, 'r', encoding='utf-8')
    label_dict = {}
    label_dict_len = 0
    while True:
      line = file.readline()
      if not line:
        break
      datas = line.split("\t")

      ###token = self.tokenizer.tokenize(datas[1])
      ###index_of_words = self.tokenizer.convert_tokens_to_ids(token)
      index_of_words = self.tokenizer.encode(datas[1])
      token_type_ids = [0] * len(index_of_words)
      attention_mask = [1] * len(index_of_words)

      # Padding Length
      padding_length = max_seq_len - len(index_of_words)

      # Zero Padding
      index_of_words += [0] * padding_length
      token_type_ids += [0] * padding_length
      attention_mask += [0] * padding_length


      label = label_dict.get(datas[0])

      if label == None:
        label=label_dict[datas[0]] = label_dict_len
        label_dict_len = label_dict_len +1
        print(label)
        print(datas[0])
      # Label
      data = {
              'input_ids': torch.tensor(index_of_words).to(self.device),
              'token_type_ids': torch.tensor(token_type_ids).to(self.device),
              'attention_mask': torch.tensor(attention_mask).to(self.device),
              'labels': torch.tensor(label).to(self.device)
             }

      self.data.append(data)

    file.close()

  def __len__(self):
    return len(self.data)
  def __getitem__(self,index):
    item = self.data[index]
    return item
