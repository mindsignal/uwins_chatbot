import math
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
model_name = "klue/roberta-base"
train_batch_size = 16
num_epochs = 4
model_save_path = "output/training_klue_sts_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


embedding_model = models.Transformer(model_name)

pooler = models.Pooling(
    embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)

model = SentenceTransformer(modules=[embedding_model, pooler])

datasets = load_dataset("klue", "sts")
testsets = load_dataset("kor_nlu", "sts")

train_samples = []
dev_samples = []
test_samples = []

# KLUE STS 내 훈련, 검증 데이터 예제 변환
for phase in ["train", "validation"]:
    examples = datasets[phase]

    for example in examples:
        score = float(example["labels"]["label"]) / 5.0  # 0.0 ~ 1.0 스케일로 유사도 정규화

        inp_example = InputExample(
            texts=[example["sentence1"], example["sentence2"]],
            label=score,
        )

        if phase == "validation":
            dev_samples.append(inp_example)
        else:
            train_samples.append(inp_example)

# KorSTS 내 테스트 데이터 예제 변환
for example in testsets["test"]:
    score = float(example["score"]) / 5.0

    if example["sentence1"] and example["sentence2"]:
        inp_example = InputExample(
            texts=[example["sentence1"], example["sentence2"]],
            label=score,
        )

    test_samples.append(inp_example)

train_samples[0].texts, train_samples[0].label
test_samples[0].texts, test_samples[0].label

train_dataloader = DataLoader(
    train_samples,
    shuffle=True,
    batch_size=train_batch_size,
)
train_loss = losses.CosineSimilarityLoss(model=model)

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    dev_samples,
    name="sts-dev",
)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1)  # 10% of train data for warm-up
logging.info(f"Warmup-steps: {warmup_steps}")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
)