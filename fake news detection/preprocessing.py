import argparse
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import subprocess
import nltk
import time
nltk.download('vader_lexicon')

def preprocess_data_initial(fake_path, true_path, output_path, neg_threshold, sim_threshold):
    df1 = pd.read_csv(fake_path)
    df2 = pd.read_csv(true_path)
    df1["label"] = "FAKE"
    df2["label"] = "True"
    data = pd.concat([df1, df2], ignore_index=True)
    sia = SentimentIntensityAnalyzer()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    data["neg_score"] = data["text"].apply(lambda x: sia.polarity_scores(x)["neg"])

    def calculate_individual_similarity(texts):
        embeddings = model.encode(texts, convert_to_tensor=True)
        similarities = cosine_similarity(embeddings.cpu())
        n = similarities.shape[0]
        results = []
        for i in range(n):
            count_similar = (similarities[i] >= sim_threshold).sum() - 1
            ratio_similar = count_similar / (n - 1) if n > 1 else 0
            results.append(ratio_similar)
        return results

    data["similarity_ratio"] = data.groupby("date")["text"].transform(lambda group: calculate_individual_similarity(group.tolist()))
    data["is_negative"] = data["neg_score"] > neg_threshold
    neg_ratio = data.groupby("date")["is_negative"].mean().reset_index()
    neg_ratio.rename(columns={"is_negative": "neg_ratio"}, inplace=True)
    data = data.merge(neg_ratio, on="date", how="left")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def extract_nouns_from_docs(docs):
        for doc in docs:
            yield " ".join([token.text for token in doc if token.pos_ == "NOUN"])

    texts = data["text"].tolist()
    docs = nlp.pipe(texts, batch_size=50, n_process=4)
    data["nouns"] = list(tqdm(extract_nouns_from_docs(docs), total=len(texts)))
    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform(data["nouns"])
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)
    topics = lda.transform(dtm)
    data["topic"] = topics.argmax(axis=1)
    feature_names = vectorizer.get_feature_names_out()
    keywords_per_topic = {
        topic_idx: [feature_names[i] for i in topic.argsort()[-3:]]
        for topic_idx, topic in enumerate(lda.components_)
    }

    def extract_keywords(text, topic):
        topic_keywords = keywords_per_topic[topic]
        text_words = set(text.split())
        matching_keywords = [word for word in topic_keywords if word in text_words]
        return ", ".join(matching_keywords)

    data["filtered_keywords"] = data.apply(lambda row: extract_keywords(row["nouns"], row["topic"]), axis=1)
    data.to_csv(output_path, index=False, encoding="utf-8")

def preprocess_data_json(fake_path,output_path, neg_threshold, sim_threshold):
    file_path = f'{fake_path}'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    data = pd.DataFrame([{'label': 'Fake' if item.get('label') == 1 else 'True', 'text': item.get('text')} 
                         for item in data if 'label' in item and 'text' in item])

    sia = SentimentIntensityAnalyzer()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    data["neg_score"] = data["text"].apply(lambda x: sia.polarity_scores(x)["neg"])

    def calculate_individual_similarity(texts):
        embeddings = model.encode(texts, convert_to_tensor=True)
        similarities = cosine_similarity(embeddings.cpu())
        n = similarities.shape[0]
        results = []
        for i in range(n):
            count_similar = (similarities[i] >= sim_threshold).sum() - 1
            ratio_similar = count_similar / (n - 1) if n > 1 else 0
            results.append(ratio_similar)
        return results

    data["similarity_ratio"] = data.groupby("date")["text"].transform(lambda group: calculate_individual_similarity(group.tolist()))
    data["is_negative"] = data["neg_score"] > neg_threshold
    neg_ratio = data.groupby("date")["is_negative"].mean().reset_index()
    neg_ratio.rename(columns={"is_negative": "neg_ratio"}, inplace=True)
    data = data.merge(neg_ratio, on="date", how="left")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def extract_nouns_from_docs(docs):
        for doc in docs:
            yield " ".join([token.text for token in doc if token.pos_ == "NOUN"])



    texts = data["text"].tolist()
    docs = nlp.pipe(texts, batch_size=50, n_process=4)  # Batch processing and parallelism
    data["nouns"] = list(tqdm(extract_nouns_from_docs(docs), total=len(texts)))

    start_time = time.time()
    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform(data["nouns"])
    print(f"Step 2 (Vectorization) completed in {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)
    print(f"Step 3 (LDA Model Fitting) completed in {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    topics = lda.transform(dtm)
    data["topic"] = topics.argmax(axis=1)
    print(f"Step 4 (Topic Assignment) completed in {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    feature_names = vectorizer.get_feature_names_out()
    keywords_per_topic = {
        topic_idx: [feature_names[i] for i in topic.argsort()[-3:]]
        for topic_idx, topic in enumerate(lda.components_)
    }
    print(f"Step 5 (Keyword Extraction per Topic) completed in {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    def extract_keywords(text, topic):
        topic_keywords = keywords_per_topic[topic]
        text_words = set(text.split())
        matching_keywords = [word for word in topic_keywords if word in text_words]
        return ", ".join(matching_keywords)

    data["filtered_keywords"] = data.progress_apply(lambda row: extract_keywords(row["nouns"], row["topic"]), axis=1)
    print(f"Step 6 (Filtered Keyword Assignment) completed in {time.time() - start_time:.2f} seconds.")

    data.to_csv(output_path, index=False, encoding="utf-8")

import pandas as pd

def preprocess_data(input_path, output_path):
    data = pd.read_csv(input_path, encoding="utf-8")
    data["filtered_keywords"] = data["filtered_keywords"].fillna("")
    for prompt_type in ["prompt_1", "prompt_2", "prompt_3"]:
        data[prompt_type] = data.apply(lambda row: create_prompt(row, prompt_type), axis=1)
    data.to_csv(output_path, index=False, encoding="utf-8")

def create_prompt(row, prompt_type):
    if prompt_type == "prompt_1":
        return f"neg_ratio {row['neg_ratio']} {row['text']}"
    elif prompt_type == "prompt_2":
        return f"similarity_ratio {row['similarity_ratio']} {row['text']}"
    elif prompt_type == "prompt_3":
        keywords = row["filtered_keywords"] if pd.notna(row["filtered_keywords"]) else "no"
        return f"topic {row['topic']} {keywords.strip()} {row['text']}".strip()
    raise ValueError("Invalid prompt type")

import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", default="./archive/Fake.csv", help="Path to Fake news dataset (CSV)")
    parser.add_argument("--true", default="./archive/True.csv", help="Path to True news dataset (CSV)")
    parser.add_argument("--output", default="./archive/output.csv", help="Path to save processed dataset (CSV)")
    parser.add_argument("--neg_threshold", type=float, default=0.05, help="Threshold for negative sentiment score")
    parser.add_argument("--sim_threshold", type=float, default=0.5, help="Threshold for similarity ratio")
    
    parser.add_argument("--json", default="./archive/fake_news_reddit_cikm20.json")
    parser.add_argument("--json_output", default="./archive/json_output.csv", help="Path to save processed dataset (CSV)")
    args = parser.parse_args()
    #preprocess_data_initial(args.fake, args.true, args.output, args.neg_threshold, args.sim_threshold)
    preprocess_data_json(args.json,args.json_output, args.neg_threshold, args.sim_threshold)
if __name__ == "__main__":
    main()
