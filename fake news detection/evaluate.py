# 필요한 라이브러리 및 모듈을 임포트
import argparse
import torch
from sklearn.metrics import f1_score
from train_eval import evaluate_or_test  # 평가 함수
from model_utils import get_model, prepare_data, TextDataset  # 모델 및 데이터셋 관련 유틸리티 함수
from transformers import AutoTokenizer  # 사전 학습된 토크나이저 사용
from torch.utils.data import DataLoader  # 데이터 로딩 유틸리티
import pandas as pd  # 데이터프레임 처리
import os
from tqdm import tqdm  # 진행 상황 표시

# 외부 테스트 데이터를 준비하는 함수
def prepare_external_test_data(path, tokenizer_name, batch_size):
    data = pd.read_csv(path)  # CSV 파일 읽기
    data = data.dropna(subset=["text"])  # "text" 열의 결측값 제거
    data["text"] = data["text"].astype(str)  # "text" 열을 문자열로 변환

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  # 사전 학습된 토크나이저 로드

    # 텍스트 데이터를 토큰화하는 내부 함수
    def tokenize_texts(texts):
        tokenized_data = {"input_ids": [], "attention_mask": []}  # 입력 데이터 초기화
        for text in tqdm(texts, desc="Tokenizing Texts"):  # 텍스트 토큰화 진행 상황 표시
            encoded = tokenizer(text, truncation=True, padding="max_length", max_length=512)  # 텍스트 토큰화
            tokenized_data["input_ids"].append(encoded["input_ids"])  # 토큰 ID 추가
            tokenized_data["attention_mask"].append(encoded["attention_mask"])  # 어텐션 마스크 추가
        return tokenized_data

    texts = data['text'].tolist()  # 텍스트 열을 리스트로 변환
    encodings = tokenize_texts(texts)  # 텍스트를 토큰화

    # "label" 열이 있는 경우 레이블과 함께 데이터셋 생성, 없는 경우 기본값 0
    if "label" in data.columns:
        labels = data["label"].tolist()
        dataset = TextDataset(encodings, labels)
    else:
        dataset = TextDataset(encodings, [0] * len(texts))

    return DataLoader(dataset, batch_size=batch_size, pin_memory=True)  # 데이터로더 반환

# 테스트 데이터 로더와 모델로 평가를 수행하는 함수
def evaluate_with_test_loader(prompt_list, test_loader, model_name, device):
    results = []  # 결과를 저장할 리스트

    for prompt in prompt_list:  # 각 프롬프트에 대해 반복
        model_path = os.path.join("ckpt", f"{prompt}_best_model.pth")  # 모델 경로 설정
        print(f"Loading model for {prompt} from {model_path}")
        
        model = get_model(model_name, num_labels=1)  # 모델 생성
        model.load_state_dict(torch.load(model_path))  # 모델 가중치 로드
        model.to(device)  # 디바이스로 모델 이동
        
        print(f"Evaluating model for {prompt}...")
        test_loss, test_acc, preds, labels, _ = evaluate_or_test(model, test_loader, device, mode="Testing")  # 평가 수행
        f1 = f1_score(labels, preds, average="weighted")  # F1 점수 계산
        results.append({"Prompt": prompt, "Test Loss": test_loss, "Test Accuracy": test_acc, "F1 Score": f1})  # 결과 저장

    results_df = pd.DataFrame(results)  # 결과를 데이터프레임으로 변환
    print("\nEvaluation Results:\n")
    print(results_df)
    return results_df  # 결과 데이터프레임 반환

# 메인 실행부
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple prompts using a test loader")  # 명령줄 인자 파서 생성
    parser.add_argument("--input", default="./archive/output.csv")  # 입력 파일 경로
    parser.add_argument("--output", default="./archive/processed_output.csv")  # 출력 파일 경로
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Pretrained model name")  # 모델 이름
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loading")  # 배치 크기
    parser.add_argument("--prompt_type", type=str, default="prompt_1", help="Prompt type used in training")  # 프롬프트 타입
    parser.add_argument("--external_test", type=str, help="Path to external test dataset (optional)", default='archive/WELFake_Dataset.csv')  # 외부 테스트 데이터 경로
    args = parser.parse_args()  # 명령줄 인자 파싱

    prompt_list = ["prompt_1", "prompt_2", "prompt_3"]  # 사용할 프롬프트 리스트
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 사용 가능한 디바이스 설정
    
    if args.external_test:  # 외부 테스트 데이터가 제공된 경우
        print(f"External test dataset provided: {args.external_test}")
        test_loader = prepare_external_test_data(args.external_test, args.model_name, args.batch_size)  # 외부 데이터 준비
    else:  # 내부 데이터에서 테스트 로더 준비
        print(f"Preparing test_loader from internal dataset...")
        train_loader, val_loader, test_loader = prepare_data(
            args.output, args.input, args.prompt_type, args.batch_size, args.model_name
        )

    results_df = evaluate_with_test_loader(prompt_list, test_loader, args.model_name, device)  # 평가 수행
    results_df.to_csv("evaluation_results.csv", index=False)  # 결과를 CSV 파일로 저장
