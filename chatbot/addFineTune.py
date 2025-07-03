import torch
from torch.utils.data import Dataset # 데이터로더
from transformers import ElectraTokenizer
import sys
import os
import numpy as np
from IPython.display import display
from tqdm import tqdm
import sys
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.activations import get_activation
from transformers import (
  ElectraPreTrainedModel,
  ElectraModel,
  ElectraConfig,
  ElectraTokenizer,
  BertConfig,
  BertTokenizer
)
import pandas as pd
from transformers import (
  AdamW,
  ElectraConfig,
  ElectraTokenizer
)

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")
sys.path.append("..")
from UlsanChatbot.model.Focal_loss import focal_loss
import UlsanChatbot.options as options

class ChatbotTextClassificationDataset(Dataset):
  """Wellness Text Classification Dataset"""
  def __init__(self,
               file_path ,
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

def train(epoch, model, optimizer, train_loader, save_step, save_ckpt_path, train_step = 0):
    losses = []
    train_start_index = train_step+1 if train_step != 0 else 0
    total_train_step = len(train_loader)
    model.train()

    with tqdm(total= total_train_step, desc=f"Train({epoch})") as pbar:
        pbar.update(train_step)
        for i, data in enumerate(train_loader, train_start_index):
            optimizer.zero_grad()
            '''
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'bias_labels': batch[3],
                      'hate_labels': batch[4]}
            if self.args.model_type != 'distilkobert':
              inputs['token_type_ids'] = batch[2]
            '''
            inputs = {'input_ids': data['input_ids'],
                      'attention_mask': data['attention_mask'],
                      'labels': data['labels'],
                      "token_type_ids" : data["token_type_ids"]
                      }
            outputs = model(**inputs)
            loss = outputs[0]
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(losses):.3f})")

            if i >= total_train_step or i % save_step == 0:
                torch.save({
                    'epoch': epoch,  # 현재 학습 epoch
                    'model_state_dict': model.state_dict(),  # 모델 저장
                    'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
                    'loss': loss.item(),  # Loss 저장
                    'train_step': i,  # 현재 진행한 학습
                    'total_train_step': len(train_loader)  # 현재 epoch에 학습 할 총 train step
                }, save_ckpt_path)
    print(losses)
    return np.mean(losses)

class ElectraClassificationHead(nn.Module):
  """Head for sentence-level classification tasks."""

  def __init__(self, config, num_labels):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, 4*config.hidden_size)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.out_proj = nn.Linear(4*config.hidden_size,num_labels)

  def forward(self, features, **kwargs):
    x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
    x = self.dropout(x)
    x = self.dense(x)
    x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
    x = self.dropout(x)
    x = self.out_proj(x)
    return x

class koElectraForSequenceClassification(ElectraPreTrainedModel):
  def __init__(self,
               config,
               num_labels):
    super().__init__(config)
    self.num_labels = num_labels
    self.electra = ElectraModel(config)
    self.classifier = ElectraClassificationHead(config, num_labels)
    self.init_weights()
  def forward(
          self,
          input_ids=None,
          attention_mask=None,
          token_type_ids=None,
          position_ids=None,
          head_mask=None,
          inputs_embeds=None,
          labels=None,
          output_attentions=None,
          output_hidden_states=None,

  ):
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
        Labels for computing the sequence classification/regression loss.
        Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
        If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
        If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    discriminator_hidden_states = self.electra(
      input_ids,
      attention_mask,
      token_type_ids,
      position_ids,
      head_mask,
      inputs_embeds,
      output_attentions,
      output_hidden_states,
    )

    sequence_output = discriminator_hidden_states[0]
    logits = self.classifier(sequence_output)

    outputs = (logits,) + discriminator_hidden_states[1:]  # add hidden states and attention if they are here

    if labels is not None:
      if self.num_labels == 1:
        #  We are doing regression
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
      else:
        #loss_fct = CrossEntropyLoss()
        #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        loss = focal_loss(logits.view(-1, self.num_labels),labels.view(-1),alpha=0.5,gamma=2.0, reduction='mean', eps=5e-6, ignore_index=options.get_label_num())
      outputs = (loss,) + outputs


    return outputs  # (loss), (logits), (hidden_states), (attentions)
def FineTune(title):
    data_path = "./UlsanChatbot/data/(add)"+title+"_text_classification_train.txt"
    checkpoint_path = "./UlsanChatbot/checkpoint"
    save_ckpt_path = f"{checkpoint_path}/"+title+"-text-classification.pth"
    model_name_or_path = "monologg/koelectra-base-discriminator"

    n_epoch = 10  # Num of Epoch
    batch_size = 1  # 배치 사이즈
    ctx = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(ctx)
    save_step = 10  # 학습 저장 주기
    learning_rate = 5e-6  # Learning Rate

    # Electra Tokenizer
    tokenizer = ElectraTokenizer.from_pretrained(model_name_or_path)

    # WellnessTextClassificationDataset 데이터 로더
    dataset = ChatbotTextClassificationDataset(tokenizer=tokenizer, device=device, file_path=data_path)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    electra_config = ElectraConfig.from_pretrained(model_name_or_path)
    model = koElectraForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                               config=electra_config,
                                                               num_labels=options.get_label_num())
    model.to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    pre_epoch, pre_loss, train_step = 0, 0, 0
    if os.path.isfile(save_ckpt_path):
        checkpoint = torch.load(save_ckpt_path, map_location=device)
        pre_epoch = checkpoint['epoch']
        pre_loss = checkpoint['loss']
        train_step = checkpoint['train_step']
        total_train_step = checkpoint['total_train_step']

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"load pretrain from: {save_ckpt_path}, epoch={pre_epoch}, loss={pre_loss}")
        # best_epoch += 1

    losses = []
    offset = pre_epoch
    for step in range(n_epoch):
        print(step)
        epoch = step + offset
        loss = train(epoch, model, optimizer, train_loader, save_step, save_ckpt_path, train_step)
        losses.append(loss)

    # data
    data = {
        "loss": losses
    }
    df = pd.DataFrame(data)
    print("losses : ", df)
    display(df)