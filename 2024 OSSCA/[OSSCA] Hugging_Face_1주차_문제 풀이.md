_** 본 문서는 GPT를 활용하여 제작되었습니다._

## Transformer 라이브러리

### pipeline

gpt-🤖:   
Transformers 라이브러리에서 제공하는 pipeline은 다양한 자연어 처리(NLP) 작업을 간단하게 수행할 수 있도록 해주는 고수준의 API입니다.  
이 pipeline을 사용하면 복잡한 설정이나 모델 세부사항에 신경 쓰지 않고도 특정 NLP 작업을 쉽게 수행할 수 있습니다.

## 감성 분석 모델 - sentiment-analysis

```python
DistilBertForSequenceClassification(
  (distilbert): DistilBertModel(
    (embeddings): Embeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (transformer): Transformer(
      (layer): ModuleList(
        (0-5): 6 x TransformerBlock(
          (attention): MultiHeadSelfAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (q_lin): Linear(in_features=768, out_features=768, bias=True)
            (k_lin): Linear(in_features=768, out_features=768, bias=True)
            (v_lin): Linear(in_features=768, out_features=768, bias=True)
            (out_lin): Linear(in_features=768, out_features=768, bias=True)
          )
          (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (ffn): FFN(
            (dropout): Dropout(p=0.1, inplace=False)
            (lin1): Linear(in_features=768, out_features=3072, bias=True)
            (lin2): Linear(in_features=3072, out_features=768, bias=True)
            (activation): GELUActivation()
          )
          (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
      )
    )
  )
  (pre_classifier): Linear(in_features=768, out_features=768, bias=True)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
)
```

gpt-🤖:  Transformers 라이브러리에서 제공하는 사전 훈련된 DistilBERT 모델을 시퀀스 분류 작업에 사용하기 위한 클래스입니다. 아래는 모델의 구조를 상세히 분석한 내용입니다.


### 1. DistilBertModel
- **embeddings**: 입력 토큰을 벡터 공간에 임베딩하는 층.
  - `word_embeddings`: 토큰 ID를 768차원의 임베딩 벡터로 변환하는 층. 어휘 크기는 30522입니다.
  - `position_embeddings`: 위치 정보를 임베딩하는 층. 최대 시퀀스 길이는 512입니다.
  - `LayerNorm`: 입력을 정규화하는 층.
  - `dropout`: 드롭아웃 레이어로, 드롭아웃 확률은 0.1입니다.

- **transformer**: 6개의 Transformer 블록으로 구성된 트랜스포머 인코더.
  - 각 Transformer 블록은 다음과 같은 하위 구성 요소로 이루어집니다:
    - **attention (MultiHeadSelfAttention)**: 다중 헤드 자기 주의 메커니즘.
      - `q_lin`, `k_lin`, `v_lin`: 쿼리, 키, 값 벡터를 생성하는 선형 변환층.
      - `out_lin`: 주의 메커니즘의 출력을 생성하는 선형 변환층.
      - `dropout`: 드롭아웃 레이어, 드롭아웃 확률은 0.1입니다.
    - **sa_layer_norm**: 자기 주의 출력의 정규화를 위한 층.
    - **ffn (FeedForward Network)**: 피드 포워드 네트워크.
      - `lin1`: 첫 번째 선형 변환층, 입력 차원 768, 출력 차원 3072.
      - `lin2`: 두 번째 선형 변환층, 입력 차원 3072, 출력 차원 768.
      - `activation`: GELU 활성화 함수.
      - `dropout`: 드롭아웃 레이어, 드롭아웃 확률은 0.1입니다.
    - **output_layer_norm**: FFN 출력의 정규화를 위한 층.

### 2. Sequence Classification 추가 층
- **pre_classifier**: 분류기 전에 있는 선형 변환층, 입력 차원 768, 출력 차원 768.
- **classifier**: 실제 분류를 수행하는 선형 변환층, 입력 차원 768, 출력 차원 2 (이진 분류의 경우).
- **dropout**: 드롭아웃 레이어, 드롭아웃 확률은 0.2입니다.

### 요약
- **입력 임베딩**: 토큰 및 위치 임베딩을 통해 입력을 벡터화합니다.
- **트랜스포머 인코더**: 6개의 Transformer 블록으로 구성된 인코더를 통해 입력 벡터를 변환합니다.
- **분류기**: 트랜스포머 인코더의 출력을 사용하여 시퀀스 분류를 수행합니다. 이를 위해 두 개의 선형 변환층과 드롭아웃을 사용합니다.

이 모델은 주로 자연어 처리(NLP) 작업에서 텍스트 분류를 위해 사용되며, 전이 학습을 통해 다양한 데이터셋에 적응할 수 있습니다.


## question-answering 파이프라인의 모델

```python
DistilBertForQuestionAnswering(
  (distilbert): DistilBertModel(
    (embeddings): Embeddings(
      (word_embeddings): Embedding(28996, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (transformer): Transformer(
      (layer): ModuleList(
        (0-5): 6 x TransformerBlock(
          (attention): MultiHeadSelfAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (q_lin): Linear(in_features=768, out_features=768, bias=True)
            (k_lin): Linear(in_features=768, out_features=768, bias=True)
            (v_lin): Linear(in_features=768, out_features=768, bias=True)
            (out_lin): Linear(in_features=768, out_features=768, bias=True)
          )
          (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (ffn): FFN(
            (dropout): Dropout(p=0.1, inplace=False)
            (lin1): Linear(in_features=768, out_features=3072, bias=True)
            (lin2): Linear(in_features=3072, out_features=768, bias=True)
            (activation): GELUActivation()
          )
          (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
      )
    )
  )
  (qa_outputs): Linear(in_features=768, out_features=2, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
)
```

gpt-🤖: Transformers 라이브러리에서 제공하는 사전 훈련된 DistilBERT 모델을 질의 응답 작업에 사용하기 위한 클래스입니다.


### 1. DistilBertModel
- **embeddings**: 입력 토큰을 벡터 공간에 임베딩하는 층.
  - `word_embeddings`: 토큰 ID를 768차원의 임베딩 벡터로 변환하는 층. 어휘 크기는 28996입니다.
  - `position_embeddings`: 위치 정보를 임베딩하는 층. 최대 시퀀스 길이는 512입니다.
  - `LayerNorm`: 입력을 정규화하는 층.
  - `dropout`: 드롭아웃 레이어로, 드롭아웃 확률은 0.1입니다.

- **transformer**: 6개의 Transformer 블록으로 구성된 트랜스포머 인코더.
  - 각 Transformer 블록은 다음과 같은 하위 구성 요소로 이루어집니다:
    - **attention (MultiHeadSelfAttention)**: 다중 헤드 자기 주의 메커니즘.
      - `q_lin`, `k_lin`, `v_lin`: 쿼리, 키, 값 벡터를 생성하는 선형 변환층.
      - `out_lin`: 주의 메커니즘의 출력을 생성하는 선형 변환층.
      - `dropout`: 드롭아웃 레이어, 드롭아웃 확률은 0.1입니다.
    - **sa_layer_norm**: 자기 주의 출력의 정규화를 위한 층.
    - **ffn (FeedForward Network)**: 피드 포워드 네트워크.
      - `lin1`: 첫 번째 선형 변환층, 입력 차원 768, 출력 차원 3072.
      - `lin2`: 두 번째 선형 변환층, 입력 차원 3072, 출력 차원 768.
      - `activation`: GELU 활성화 함수.
      - `dropout`: 드롭아웃 레이어, 드롭아웃 확률은 0.1입니다.
    - **output_layer_norm**: FFN 출력의 정규화를 위한 층.

### 2. Question Answering 추가 층
- **qa_outputs**: 질의 응답을 위한 선형 변환층, 입력 차원 768, 출력 차원 2.
  - 이 선형 층은 시작 위치와 끝 위치를 예측하는 데 사용됩니다.
- **dropout**: 드롭아웃 레이어, 드롭아웃 확률은 0.1입니다.

### 요약
- **입력 임베딩**: 토큰 및 위치 임베딩을 통해 입력을 벡터화합니다.
- **트랜스포머 인코더**: 6개의 Transformer 블록으로 구성된 인코더를 통해 입력 벡터를 변환합니다.
- **질의 응답 출력**: 트랜스포머 인코더의 출력을 사용하여 텍스트 내에서 질문에 대한 답변의 시작과 끝 위치를 예측합니다.

이 모델은 주로 자연어 처리(NLP) 작업에서 질문에 대한 답변을 찾기 위해 사용되며, 전이 학습을 통해 다양한 데이터셋에 적응할 수 있습니다. 질의 응답 작업에서 입력된 질문과 문맥을 기반으로 적절한 답변의 위치를 찾아내는 역할을 합니다.


## 두 모델 비교하기

`DistilBertForQuestionAnswering`와 `DistilBertForSequenceClassification`는 둘 다 DistilBERT 기반의 모델로, Transformers 라이브러리에서 제공하는 사전 훈련된 모델입니다. 그러나 이 두 모델은 각각 다른 NLP 작업을 위해 설계되었습니다. 아래는 두 모델의 구조와 사용 사례를 비교한 내용입니다.

### 공통점
1. **기반 모델**: 두 모델 모두 DistilBERT 모델을 기반으로 합니다. 이는 경량화된 BERT 모델로, BERT의 성능을 유지하면서 더 빠르고 효율적으로 작동합니다.
2. **입력 임베딩**: 두 모델 모두 입력 토큰을 벡터로 변환하기 위해 `word_embeddings`, `position_embeddings`, `LayerNorm`, 및 `dropout` 층을 사용합니다.
3. **트랜스포머 인코더**: 두 모델 모두 6개의 Transformer 블록으로 구성된 트랜스포머 인코더를 사용합니다.

### DistilBertForQuestionAnswering
#### 구조
- **qa_outputs**: 선형 층으로, 입력 차원 768, 출력 차원 2. 이 층은 시작 위치와 끝 위치를 예측하는 데 사용됩니다.
- **dropout**: 드롭아웃 확률은 0.1입니다.

#### 사용 사례
- **질의 응답(QA)**: 주어진 문맥에서 질문에 대한 답변을 찾는 작업. 예를 들어, 문장에서 "파란 하늘이 보였다. 그가 가장 좋아하는 색은 무엇인가?"라는 질문이 주어졌을 때, 문맥에서 "파란"이라는 답변을 찾아냅니다.

#### 예시 코드
```python
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

question, text = "What is his favorite color?", "He looked at the blue sky. Blue is his favorite color."

inputs = tokenizer(question, text, return_tensors='pt')
outputs = model(**inputs)

start_scores = outputs.start_logits
end_scores = outputs.end_logits
```

### DistilBertForSequenceClassification
#### 구조
- **pre_classifier**: 선형 층으로, 입력 차원 768, 출력 차원 768.
- **classifier**: 선형 층으로, 입력 차원 768, 출력 차원은 분류할 클래스의 수 (예: 이진 분류의 경우 2).
- **dropout**: 드롭아웃 확률은 0.2입니다.

#### 사용 사례
- **텍스트 분류**: 텍스트를 주어진 카테고리로 분류하는 작업. 예를 들어, 영화 리뷰를 긍정적 또는 부정적으로 분류하는 작업에 사용됩니다.

#### 예시 코드
```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

inputs = tokenizer("This movie is great!", return_tensors='pt')
outputs = model(**inputs)

logits = outputs.logits
```

### 주요 차이점 요약
- **목적**: `DistilBertForQuestionAnswering`는 질문에 대한 답변을 찾기 위해 사용되고, `DistilBertForSequenceClassification`는 텍스트를 특정 카테고리로 분류하는 데 사용됩니다.
- **출력 구조**: `DistilBertForQuestionAnswering`는 시작 위치와 끝 위치를 예측하는 선형 층을 가지고 있으며, `DistilBertForSequenceClassification`는 텍스트를 분류하는 선형 층을 가지고 있습니다.
- **드롭아웃 확률**: `DistilBertForQuestionAnswering`는 드롭아웃 확률이 0.1이고, `DistilBertForSequenceClassification`는 0.2입니다.

이 두 모델은 각기 다른 NLP 작업에 특화되어 있으므로, 작업의 특성에 따라 적절한 모델을 선택하여 사용할 수 있습니다.

## Note. 

- Q. transformer pipeline에 내가 원하는 모델을 연결할 수 있을까? 
  - (model hub에 없고, 내가 가지고 있는 계정의 GPT를 연결할 수 있을까?)
- DistilBertForQuestionAnswering와 DistilBertForSequenceClassification 두 모델이 비슷하지만, 출력 층 부분이 다른 점이 흥미로웠다.


