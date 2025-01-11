# 필요한 라이브러리 및 모듈을 임포트
import pandas as pd
import torch
from sklearn.model_selection import train_test_split  # 데이터 분할을 위한 유틸리티
from torch.utils.data import DataLoader  # 데이터 로딩 유틸리티
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # 사전 학습된 토크나이저와 모델

# 텍스트 데이터셋 클래스 정의
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  # 입력 데이터를 저장
        self.labels = labels  # 레이블을 저장

    def __len__(self):
        return len(self.labels)  # 데이터셋 크기 반환

    def __getitem__(self, idx):
        # 인덱스에 해당하는 데이터를 딕셔너리 형태로 반환
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])  # 레이블 추가
        return item

# 데이터 준비 함수
def prepare_data(train_path, test_path, prompt_type, batch_size, tokenizer_name):
    # 학습 데이터 읽기
    data = pd.read_csv(train_path, encoding="utf-8")
    train_val, _ = train_test_split(data, test_size=0.2, random_state=42)  # train-validation 분리
    train, _ = train_test_split(train_val, test_size=0.25, random_state=42)  # train 분리

    # 테스트 데이터 읽기 및 분할
    data_test = pd.read_csv(test_path, encoding="utf-8")
    train_val, test = train_test_split(data_test, test_size=0.2, random_state=42)
    _, val = train_test_split(train_val, test_size=0.25, random_state=42)

    # 사전 학습된 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 텍스트 데이터를 토큰화하는 함수
    def tokenize_texts(texts):
        return tokenizer(texts, truncation=True, padding=True, max_length=256)  # 토큰화 수행

    # 데이터셋 생성 함수
    def create_dataset(data, encodings, label_column):
        labels = [label_mapping[label] for label in data[label_column]]  # 레이블 매핑
        return TextDataset(encodings, labels)

    # 레이블 매핑 정의
    label_mapping = {"FAKE": 0, "True": 1}

    # 텍스트와 레이블 준비
    train_texts = train[prompt_type].tolist()  # 학습 텍스트
    val_texts = val['text'].tolist()  # 검증 텍스트
    test_texts = test['text'].tolist()  # 테스트 텍스트

    # 텍스트 토큰화
    train_encodings = tokenize_texts(train_texts)
    val_encodings = tokenize_texts(val_texts)
    test_encodings = tokenize_texts(test_texts)

    # 데이터셋 생성
    train_dataset = create_dataset(train, train_encodings, "label")
    val_dataset = create_dataset(val, val_encodings, "label")
    test_dataset = create_dataset(test, test_encodings, "label")

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

    # 학습, 검증, 테스트 데이터 로더 반환
    return train_loader, val_loader, test_loader

# 모델 생성 함수
def get_model(model_name, num_labels):
    # 사전 학습된 모델 로드 및 출력 레이블 수 설정
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
