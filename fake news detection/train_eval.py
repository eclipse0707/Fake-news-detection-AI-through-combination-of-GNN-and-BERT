import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np

def train_one_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    loss_fn = torch.nn.BCEWithLogitsLoss()  # BCE 손실 함수 사용

    for batch in tqdm(train_loader, desc="Training"):
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device)
        }
        labels = batch["labels"].to(device).float()  # BCE 손실의 경우 float 형으로 변환 필요 (0, 1)
        optimizer.zero_grad()
        outputs = model(**inputs)

        # 모델 출력이 1차원인 경우 그대로 사용
        logits = outputs.logits.squeeze(-1)  # 1차원으로 변환 (BCE에 적합)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        # 예측 값 계산 (시그모이드 적용 후 0.5 기준)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), correct / total

def evaluate_or_test(model, dataloader, device, mode="Evaluation"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    loss_fn = torch.nn.BCEWithLogitsLoss()  # BCE 손실 함수 사용

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=mode):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            labels = batch["labels"].to(device).float()  # BCE 손실의 경우 float 형으로 변환 필요 (0, 1)
            outputs = model(**inputs)

            # 모델 출력이 1차원인 경우 그대로 사용
            logits = outputs.logits.squeeze(-1)  # 1차원으로 변환 (BCE에 적합)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            # 예측 값 계산 (시그모이드 적용 후 0.5 기준)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)
