from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
# Create your views here.
from queue import PriorityQueue
import random
import sys
import os
import torch

from transformers import (
  ElectraConfig,
  ElectraTokenizer
)

sys.path.append("..")
from chatbot.answers import *
from chatbot.addFineTune import FineTune

from UlsanChatbot.model.koelectra import koElectraForSequenceClassification,koelectra_input
import UlsanChatbot.options as options

def load_chatbot_answer(title):
  root_path = './UlsanChatbot'
  file_path = f"{root_path}/data/"+title+"_text_classification_train.txt"
  f_f = open(file_path,'r',encoding='utf-8')
  file_lines = f_f.readlines()
  category = {}
  answer = {}
  k = 0
  for data in file_lines:
    data = data.split('\t')
    category_values = category.values()
    if data[0] in category_values:
      pass
    else:
      category[str(k)] = data[0]
      k = k+1
    answer_keys = answer.keys()
    if data[0] in answer_keys:
      if len(data) > 2:
        if len(data[2][:-1]) > 0:
          answer[data[0]] += [data[2][:-1]]
          print(answer[data[0]])
    else:
      if len(data) > 2:
        answer[data[0]] = [data[2][:-1]]
  return category, answer

root_path='./UlsanChatbot'
checkpoint_path =f"{root_path}/checkpoint"
save_ckpt_path = f"{checkpoint_path}/hubok-text-classification.pth"
model_name_or_path = "monologg/koelectra-base-discriminator"

#답변과 카테고리 불러오기
category, answer = load_chatbot_answer("hubok")

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

# 저장한 Checkpoint 불러오기

checkpoint = torch.load(save_ckpt_path, map_location=device)

# Electra Tokenizer
tokenizer = ElectraTokenizer.from_pretrained(model_name_or_path)

electra_config = ElectraConfig.from_pretrained(model_name_or_path)
model = koElectraForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                             config=electra_config,
                                                             num_labels=options.get_label_num())
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()

############################################################################
root_path2='./UlsanChatbot'
checkpoint_path2 =f"{root_path2}/checkpoint"
save_ckpt_path2 = f"{checkpoint_path2}/deglok-text-classification.pth"
model_name_or_path2 = "monologg/koelectra-base-discriminator"

#답변과 카테고리 불러오기
category2, answer2 = load_chatbot_answer("deglok")

ctx = "cuda" if torch.cuda.is_available() else "cpu"
#ctx = "cpu"
device = torch.device(ctx)

# 저장한 Checkpoint 불러오기

checkpoint2 = torch.load(save_ckpt_path2, map_location=device)

# Electra Tokenizer
tokenizer2 = ElectraTokenizer.from_pretrained(model_name_or_path2)

electra_config2 = ElectraConfig.from_pretrained(model_name_or_path2)
model2 = koElectraForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path2,
                                                             config=electra_config2,
                                                             num_labels=options.get_label_num())
model2.load_state_dict(checkpoint2['model_state_dict'])

model2.to(device)
model2.eval()



class Log:
    def __init__(self):
        delok_count_log = open("delok_count_log.txt", "r")
        hubok_count_log = open("hubok_count_log.txt", "r")
        self.delok_count = int(str(delok_count_log.read()))
        self.hubok_count = int(str(hubok_count_log.read()))
        delok_count_log.close()
        hubok_count_log.close()

    def add_count_hubok(self):
        self.hubok_count = self.hubok_count + 1
        log_file = open("hubok_count_log.txt", "w")
        if (self.hubok_count < 1000):
            log_file.write(str(self.hubok_count))
        else:
            self.hubok_count = 0
            log_file.write(str(self.hubok_count))
            print("fine tuning 시작")
            FineTune(hubok)
    def add_count_delok(self):
        self.delok_count = self.delok_count + 1
        log_file = open("delok_count_log.txt", "w")
        if (self.delok_count < 1000):
            log_file.write(str(self.delok_count))
        else:
            self.delok_count = 0
            log_file.write(str(self.delok_count))
            FineTune(delok)
log = Log()
def home(request):
    context = {}
    return render(request, "chathome.html", context)

def delok(request):
    context={}
    return render(request, "delok.html", context)

def hubok(request):
    context={}
    return render(request,"hubok.html", context)
def sugang(request):
    context={}
    return render(request,"sugang.html", context)
def welfare(request):
    context={}
    return render(request,"welfare.html", context)

def chatanswer_hubok_correct(request):
    context = {}
    chattext = request.GET["chattext"]
    label = request.GET["label"]
    f = open("UlsanChatbot/data/(add)hubok_text_classification_train.txt","a", encoding="utf-8")
    f.write(str(label)+"\t"+str(chattext)+"\n")
    f.close()
    log.add_count_hubok()
    return JsonResponse(context, content_type="application/json")

def chatanswer_delok_correct(request):
    context = {}
    chattext = request.GET["chattext"]
    label = request.GET["label"]
    f = open("UlsanChatbot/data/(add)delok_text_classification_train.txt","a", encoding="utf-8")
    f.write(str(label)+"\t"+str(chattext)+"\n")
    f.close()
    log.add_count_delok()
    return JsonResponse(context, content_type="application/json")
def chatanswer_hubok_incorrect(request):
    context = {}
    max_index = hubok_answer_queue.get()[1]
    max_index_value = softmax_logit_hubok[max_index]
    answer_list = answer[category[str(max_index)]]
    answer_len = len(answer_list) - 1
    answer_index = random.randint(0, answer_len)
    context["result"] = answer_list[answer_index]
    context["category"] = category[str(max_index)]
    print(f'Answer: {answer_list[answer_index]}, index: {max_index}, softmax_value: {max_index_value}')
    print('-' * 50)
    return JsonResponse(context, content_type="application/json")

def chatanswer_delok_incorrect(request):
    context = {}
    max_index = delok_answer_queue.get()[1]
    max_index_value = softmax_logit_delok[max_index]
    answer_list = answer2[category2[str(max_index)]]
    answer_len = len(answer_list) - 1
    answer_index = random.randint(0, answer_len)
    context["result"] = answer_list[answer_index]
    context["category"] = category2[str(max_index)]
    print(f'Answer: {answer_list[answer_index]}, index: {max_index}, softmax_value: {max_index_value}')
    print('-' * 50)
    return JsonResponse(context, content_type="application/json")



@csrf_exempt
def chatanswer_hubok(request):
     context = {}
     global hubok_answer_queue
     hubok_answer_queue = PriorityQueue()
     sent = request.GET["chattext"]
     data = koelectra_input(tokenizer, sent, device, 512)
     output = model(**data)
     logit = output[0]
     global softmax_logit_hubok
     softmax_logit_hubok = torch.softmax(logit, dim=1)
     softmax_logit_hubok = softmax_logit_hubok.squeeze()
     softmax_logit_hubok = softmax_logit_hubok.cpu().detach().numpy()
     for index, value in enumerate(softmax_logit_hubok):
         hubok_answer_queue.put((-value, index))
     max_index = hubok_answer_queue.get()[1]
     max_index_value = softmax_logit_hubok[max_index]
     answer_list = answer[category[str(max_index)]]
     answer_len = len(answer_list) - 1
     answer_index = random.randint(0, answer_len)
     context["result"] = answer_list[answer_index]
     context["category"] = category[str(max_index)]
     print(f'Answer: {answer_list[answer_index]}, index: {max_index}, softmax_value: {max_index_value}')
     print('-' * 50)
     return JsonResponse(context, content_type="application/json")

def chatanswer_delok(request):
    global delok_answer_queue
    delok_answer_queue = PriorityQueue()
    context={}
    sent = request.GET["chattext"]
    
    data = koelectra_input(tokenizer2, sent, device, 512)
    output = model2(**data)
    logit = output[0]
    global softmax_logit_delok
    softmax_logit_delok = torch.softmax(logit, dim=1)
    softmax_logit_delok = softmax_logit_delok.squeeze()
    softmax_logit_delok = softmax_logit_delok.cpu().detach().numpy()
    for index, value in enumerate(softmax_logit_delok):
        delok_answer_queue.put((-value, index))
    max_index = delok_answer_queue.get()[1]
    max_index_value = softmax_logit_delok[max_index]
    answer_list = answer2[category2[str(max_index)]]
    answer_len = len(answer_list) - 1
    answer_index = random.randint(0, answer_len)
    print(f'Answer: {answer_list[answer_index]}, index: {max_index}, softmax_value: {max_index_value}')
    print('-' * 50)
    context["result"] = answer_list[answer_index]
    context["category"] = category2[str(max_index)]
    return JsonResponse(context, content_type="application/json")

def unknown_question(request):
    sent = request.GET["chattext"]
    f = open("UlsanChatbot/data/unknown_question.txt","a", encoding="utf-8")
    f.write(sent+"\n")
    f.close()
    return JsonResponse({}, content_type="application/json")

def chatanswer_sugang(request):
    context={}
    sent = request.GET["chattext"]
    context["result"] = descision_sugang(sent)
    if context["result"] != -1:
        return JsonResponse(context, content_type="application/json")
    else:
        context["result"] = "teset"
        return JsonResponse(context, content_type="application/json")

def chatanswer_welfare(request):
    context={}
    sent = request.GET["chattext"]
    context["result"] = descision_welfare(sent)
    if context["result"] != -1:
        return JsonResponse(context, content_type="application/json")
    else:
        context["result"] = "test"
        return JsonResponse(context, content_type="application/json")