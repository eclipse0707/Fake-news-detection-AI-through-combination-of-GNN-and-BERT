# 필요한 라이브러리 및 모듈을 임포트
import argparse
import torch
import random
import numpy as np
from transformers import AdamW, get_scheduler  # 옵티마이저 및 학습 스케줄러
from preprocessing import preprocess_data  # 데이터 전처리 함수
from model_utils import prepare_data, get_model, TextDataset  # 모델 및 데이터셋 관련 유틸리티 함수
from train_eval import train_one_epoch, evaluate_or_test  # 학습 및 평가 함수
from sklearn.metrics import f1_score  # 평가 지표
import pandas as pd  # 데이터프레임 처리
from transformers import AutoTokenizer  # 사전 학습된 토크나이저
from torch.utils.data import DataLoader  # 데이터 로딩 유틸리티
import os
from tqdm import tqdm  # 진행 상황 표시
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 라이브러리 충돌 문제 해결

# 시드 고정 함수 (결과 재현성을 위한 함수)
def set_seed(seed):
    random.seed(seed)  # 파이썬의 랜덤 시드 고정
    np.random.seed(seed)  # NumPy의 랜덤 시드 고정
    torch.manual_seed(seed)  # PyTorch의 CPU 시드 고정
    torch.cuda.manual_seed(seed)  # PyTorch의 GPU 시드 고정
    torch.cuda.manual_seed_all(seed)  # 모든 GPU의 시드 고정
    torch.backends.cudnn.deterministic = True  # CUDNN의 결정론적 동작 설정
    torch.backends.cudnn.benchmark = False  # CUDNN 최적화 비활성화

# 외부 테스트 데이터를 준비하는 함수
def prepare_external_test_data(path, prompt_type, tokenizer_name, batch_size, label_mapping=None):
    # CSV 파일 읽기
    data = pd.read_csv(r'C:\Users\user\Desktop\진지한개미8077\archive\WELFake_Dataset.csv')
    data = data.dropna(subset=["text"])  # "text" 열에서 결측값 제거
    data["text"] = data["text"].astype(str)  # "text" 열을 문자열로 변환

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  # 사전 학습된 토크나이저 로드

    # 텍스트 데이터를 토큰화하는 함수
    def tokenize_texts(texts):
        tokenized_data = {"input_ids": [], "attention_mask": []}  # 입력 데이터 초기화
        for text in tqdm(texts, desc="Tokenizing Texts"):  # 진행 상황 표시
            encoded = tokenizer(text, truncation=True, padding="max_length", max_length=512)  # 토큰화 수행
            tokenized_data["input_ids"].append(encoded["input_ids"])  # 토큰 ID 추가
            tokenized_data["attention_mask"].append(encoded["attention_mask"])  # 어텐션 마스크 추가
        return tokenized_data

    texts = data['text'].tolist()  # 텍스트 열을 리스트로 변환
    encodings = tokenize_texts(texts)  # 텍스트를 토큰화

    # 레이블 존재 여부 확인 후 데이터셋 생성
    if "label" in data.columns:
        labels = data["label"].tolist()
        dataset = TextDataset(encodings, labels)  # 레이블과 함께 데이터셋 생성
    else:
        dataset = TextDataset(encodings, [0] * len(texts))  # 레이블 없는 경우 기본값 사용

    return DataLoader(dataset, batch_size=batch_size, pin_memory=True)  # 데이터로더 반환

# 메인 함수
def main():
    # 명령줄 인자 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./archive/output.csv")  # 입력 파일 경로
    parser.add_argument("--output", default="./archive/processed_output.csv")  # 출력 파일 경로
    parser.add_argument("--model_name", default="distilbert-base-uncased")  # 사전 학습된 모델 이름
    parser.add_argument("--num_epochs", type=int, default=10)  # 에폭 수
    parser.add_argument("--batch_size", type=int, default=32)  # 배치 크기
    parser.add_argument("--prompt_type", default="prompt_1")  # 프롬프트 유형
    parser.add_argument("--external_test", help="Path to external test dataset (optional)", default="None")  # 외부 테스트 경로
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")  # 랜덤 시드
    args = parser.parse_args()

    # 랜덤 시드 고정
    set_seed(args.seed)

    # 체크포인트 폴더 생성
    os.makedirs("ckpt", exist_ok=True)
    best_model_path = os.path.join("ckpt", f"{args.prompt_type}_best_model.pth")  # 모델 저장 경로

    print(args.external_test)
    if args.external_test != "None":  # 외부 테스트 데이터가 제공된 경우
        print("External test dataset provided. Skipping training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 사용 가능한 디바이스 설정
        model = get_model(args.model_name, num_labels=1)  # 모델 생성
        model.load_state_dict(torch.load(best_model_path))  # 저장된 모델 로드
        model.to(device)  # 디바이스로 모델 이동
        print(f"Loaded best model from {best_model_path}")

        test_loader = prepare_external_test_data(
            args.external_test, args.prompt_type, args.model_name, args.batch_size
        )  # 외부 테스트 데이터 준비
        print(f"Start Testing")
        test_loss, test_acc, preds, labels, probs = evaluate_or_test(model, test_loader, device, mode="Testing")  # 테스트 수행
        f1 = f1_score(labels, preds, average="weighted")  # F1 점수 계산
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f} Test F1 Score: {f1:.4f}")  # 테스트 결과 출력
        return

    else:  # 학습 및 검증 수행
        preprocess_data(args.input, args.output)  # 데이터 전처리
        train_loader, val_loader, test_loader = prepare_data(
            args.output, args.input, args.prompt_type, args.batch_size, args.model_name
        )  # 데이터 로더 준비
        model = get_model(args.model_name, num_labels=1)  # 모델 생성
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 디바이스 설정
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=5e-5)  # AdamW 옵티마이저 설정
        scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * args.num_epochs
        )  # 학습 스케줄러 설정

        best_val_loss = float("inf")  # 가장 낮은 검증 손실 초기화

        for epoch in range(args.num_epochs):  # 에폭 반복
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)  # 한 에폭 학습
            print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Accuracy={train_acc:.4f}")  # 학습 결과 출력
            val_loss, val_acc, _, _, _ = evaluate_or_test(model, val_loader, device, mode="Validation")  # 검증 수행
            print(f"Epoch {epoch + 1}: Val Loss={val_loss:.4f}, Val Accuracy={val_acc:.4f}")  # 검증 결과 출력

            if val_loss < best_val_loss:  # 검증 손실이 개선된 경우
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)  # 모델 저장
                print(f"Best model saved with Val Loss={val_loss:.4f} at epoch {epoch + 1}")  # 저장 메시지 출력

if __name__ == "__main__":
    main()  # 메인 함수 실행
