# NLP_2025: 고급 자연어 처리 예제

이 저장소는 모델 미세 조정, 텍스트 생성, 배포 등 다양한 고급 자연어 처리(NLP) 기술을 보여주는 Python 스크립트를 포함하고 있습니다.

## 프로젝트 개요

이 프로젝트는 다음 내용을 탐구합니다:
- 특정 작업(예: 감성 분석)을 위한 사전 학습된 트랜스포머 모델(BERT) 미세 조정.
- 자기 회귀 모델(GPT-2)을 사용한 일관성 있는 텍스트 생성.
- 최적화된 추론을 위한 PyTorch 모델의 ONNX 형식 변환.
- FastAPI 및 Docker를 사용한 NLP 모델의 웹 서비스 배포.

## 프로젝트 구조

```
NLP_2025/
├── .cache/                     # Hugging Face Datasets 캐시 디렉토리
├── .git/                       # Git 버전 관리 디렉토리
├── logs_bert_finetune_07/      # BERT 미세 조정 시 생성되는 로그 디렉토리
├── onnx_models/                # 변환된 ONNX 모델 저장 디렉토리
│   └── bert_classifier_nsmc_finetuned.onnx
├── results_bert_finetune_07/   # BERT 미세 조정 결과물(체크포인트) 저장 디렉토리
│   └── checkpoint-XX/
├── 01_setup_and_data.py        # 환경 설정 및 데이터 준비 스크립트
├── 02_rnn_classification.py    # RNN 기반 텍스트 분류 스크립트
├── 02_rnn_classification_history.png # RNN 학습 결과 그래프
├── 03_lstm_gru_comparison.py   # LSTM/GRU 비교 스크립트
├── 04_seq2seq_basics.py        # Seq2Seq 기본 스크립트
├── 05_attention_mechanisms.py  # 어텐션 메커니즘 스크립트
├── 06_transformer_from_scratch.py # 트랜스포머 직접 구현 스크립트
├── 07_finetune_bert.py         # BERT 미세 조정 스크립트
├── 08_gpt_text_generation.py   # GPT-2 텍스트 생성 스크립트
├── 09_deployment_api_onnx.py   # ONNX 변환 및 배포 준비 스크립트
├── Dockerfile                  # Docker 이미지 빌드 파일
├── main.py                     # FastAPI 애플리케이션 실행 스크립트
├── README.md                   # 프로젝트 설명 파일
└── requirements.txt            # Python 의존성 목록 파일
```

## 환경 설정

Conda 환경 사용을 권장합니다. 이 스크립트 개발에 사용된 환경은 Python 3.9 기반의 `colab_nlp_env`입니다.

1.  **Conda 환경 생성 (이미 생성하지 않은 경우):**
    ```bash
    conda create -n colab_nlp_env python=3.9
    conda activate colab_nlp_env
    ```

2.  **의존성 설치:**
    주요 의존성 패키지: `transformers`, `torch`, `datasets`, `scikit-learn`, `onnx`, `onnxruntime`, `fastapi`, `uvicorn`.
    배포 API를 위한 `requirements.txt` 파일이 제공되며, 일반적인 의존성 참조용으로도 사용할 수 있습니다.
    ```bash
    # 배포 API용 (및 일반 사용)
    pip install -r requirements.txt

    # 특정 스크립트에 추가 패키지가 필요할 수 있습니다 (예: spacy).
    # pip install spacy
    # python -m spacy download ko_core_news_sm # Spacy를 사용한 한국어 토큰화 시
    ```

## 스크립트 설명

### 1. `01_setup_and_data.py` - 환경 설정 및 데이터 준비

-   **설명:** NLP 실습에 필요한 기본 환경을 설정하고 데이터를 준비합니다.
-   **주요 기능:**
    -   필수 패키지 설치 언급 (주석 처리됨).
    -   Hugging Face `datasets` 라이브러리를 사용하여 샘플 데이터셋(IMDb)을 로드하고 구조를 확인합니다.
    -   BPE (GPT-2에서 사용) 및 WordPiece (BERT에서 사용) 토큰화 방법을 비교합니다.
-   **사용법:**
    ```bash
    python 01_setup_and_data.py
    ```

### 2. `02_rnn_classification.py` - RNN을 이용한 텍스트 분류

-   **설명:** 기본적인 순환 신경망(RNN) 모델을 구현하여 텍스트 분류(IMDb 감성 분석) 작업을 수행합니다.
-   **주요 기능:**
    -   IMDb 데이터셋을 로드하고 토큰화합니다.
    -   `nn.Embedding`, `nn.RNN`, `nn.Linear`를 사용하여 간단한 RNN 분류기를 정의합니다.
    -   학습 루프를 포함하며, 손실 및 정확도를 계산합니다.
    -   학습 과정(손실 및 정확도)을 시각화하고 저장합니다.
-   **사용법:**
    ```bash
    python 02_rnn_classification.py
    ```

### 3. `03_lstm_gru_comparison.py` - LSTM 및 GRU 비교

-   **설명:** RNN의 장기 의존성 문제를 개선한 LSTM(Long Short-Term Memory) 및 GRU(Gated Recurrent Unit) 모델을 구현하고 RNN과 성능을 비교합니다.
-   **주요 기능:**
    -   RNN 외에 LSTM 및 GRU 분류기를 정의하고 구현합니다.
    -   IMDb 감성 분류 작업에 대해 세 가지 모델(RNN, LSTM, GRU)을 각각 학습시킵니다.
    -   정확도 및 학습 시간 측면에서 모델들의 성능을 비교합니다.
-   **사용법:**
    ```bash
    python 03_lstm_gru_comparison.py
    ```

### 4. `04_seq2seq_basics.py` - Seq2Seq 기본

-   **설명:** 시퀀스-투-시퀀스(Seq2Seq) 아키텍처의 기본 구조를 소개합니다.
-   **주요 기능:**
    -   IWSLT2017 영어-프랑스어 번역 데이터셋의 작은 부분집합을 로드합니다.
    -   Spacy를 사용하여 토큰화하고 소스(영어) 및 타겟(프랑스어) 언어에 대한 사용자 정의 어휘집을 구축합니다.
    -   `EncoderRNN` (LSTM 기반) 및 `DecoderRNN` (LSTM 기반)을 정의합니다.
    -   인코더와 디코더를 래핑하고 기본적인 형태의 교사 강요(teacher forcing)를 포함하는 `Seq2Seq` 모델을 구현합니다.
    -   모델을 초기화하지만, 전체 학습 루프는 포함하지 않고 구조적 구성 요소에 중점을 둡니다.
-   **사용법:**
    ```bash
    python 04_seq2seq_basics.py
    ```

### 5. `05_attention_mechanisms.py` - 어텐션 메커니즘

-   **설명:** Seq2Seq 모델의 성능을 향상시키는 어텐션 메커니즘을 다룹니다.
-   **주요 기능:**
    -   `04_seq2seq_basics.py`의 데이터 로딩 및 어휘집 설정을 재사용합니다.
    -   Bahdanau (additive) 어텐션을 별도의 모듈로 구현합니다.
    -   Bahdanau 어텐션 메커니즘과 GRU를 통합한 `AttnDecoderRNN`을 정의합니다.
    -   어텐션의 구현 세부 사항에 중점을 두며, Luong 어텐션 구현 및 성능 비교를 연습 문제로 제안합니다. 전체 학습 루프는 포함하지 않습니다.
-   **사용법:**
    ```bash
    python 05_attention_mechanisms.py
    ```

### 6. `06_transformer_from_scratch.py` - 트랜스포머 직접 구현

-   **설명:** 트랜스포머 아키텍처의 핵심 구성 요소를 처음부터 직접 구현합니다.
-   **주요 기능:**
    -   `ScaledDotProductAttention`을 구현합니다.
    -   스케일드 닷-프로덕트 어텐션을 사용하여 `MultiHeadAttention`을 구축합니다.
    -   멀티-헤드 셀프 어텐션, 피드포워드 네트워크, 레이어 정규화, 드롭아웃을 포함하는 기본적인 `EncoderLayer`를 정의합니다.
    -   기본적인 구성 요소를 설명하고, 위치 인코딩, 디코더 레이어 구현 및 전체 트랜스포머 모델 조립을 연습 문제로 제안합니다.
-   **사용법:**
    ```bash
    python 06_transformer_from_scratch.py
    ```

### 7. `07_finetune_bert.py` - BERT 미세 조정 (감성 분석)

-   **설명:** 사전 학습된 BERT 모델(예: `klue/bert-base`)을 NSMC(네이버 영화리뷰 감성분석) 데이터셋을 사용하여 감성 분석 작업에 맞게 미세 조정합니다.
-   **주요 기능:**
    -   NSMC 데이터셋을 로드하고 전처리합니다.
    -   Hugging Face `Trainer` API를 사용하여 미세 조정을 수행합니다.
    -   모델을 평가하고 미세 조정된 모델과 토크나이저를 저장합니다.
-   **출력:** 미세 조정된 모델과 토크나이저는 `./results_bert_finetune_07/checkpoint-XX/`에 저장됩니다.
-   **사용법:**
    ```bash
    python 07_finetune_bert.py
    ```

### 8. `08_gpt_text_generation.py` - GPT-2 텍스트 생성

-   **설명:** 사전 학습된 한국어 GPT-2 모델(예: `skt/kogpt2-base-v2`)을 사용하여 텍스트 생성을 시연합니다.
-   **주요 기능:**
    -   GPT-2 모델과 토크나이저를 로드합니다.
    -   다양한 샘플링 매개변수(temperature, top-k, top-p)를 사용한 텍스트 생성 예제를 보여줍니다.
    -   프롬프트 기반 텍스트 요약 예제를 포함합니다.
-   **사용법:**
    ```bash
    python 08_gpt_text_generation.py
    ```

### 9. `09_deployment_api_onnx.py` - 모델 ONNX 변환 및 배포 준비

-   **설명:**
    -   미세 조정된 BERT 모델(`07_finetune_bert.py`에서 생성)을 로드합니다.
    -   PyTorch 모델을 최적화된 추론을 위해 ONNX 형식으로 변환합니다.
    -   ONNX 모델의 출력을 원본 PyTorch 모델과 비교하여 검증합니다.
    -   배포를 위한 FastAPI 서버, `Dockerfile`, `requirements.txt` 예제 코드를 출력합니다 (이 파일들은 현재 `main.py`, `Dockerfile`, `requirements.txt`로 별도 생성됨).
-   **출력:**
    -   ONNX 모델은 `./onnx_models/bert_classifier_nsmc_finetuned.onnx`에 저장됩니다.
-   **사용법:**
    ```bash
    python 09_deployment_api_onnx.py
    ```

## 미세 조정된 BERT 모델 배포 (FastAPI 및 Docker 사용)

이 프로젝트는 미세 조정된 감성 분석 BERT 모델을 웹 API로 배포하기 위한 파일들을 포함합니다.

-   `main.py`: ONNX 모델을 서빙하는 FastAPI 애플리케이션입니다.
-   `Dockerfile`: Docker 이미지를 빌드하기 위한 파일입니다.
-   `requirements.txt`: API를 위한 Python 의존성 목록입니다.

### Docker 컨테이너 빌드 및 실행

Docker가 설치되어 실행 중인지 확인하십시오. 프로젝트 루트 디렉토리 (`/home/kwy00/projects/NLP_2025/`)로 이동합니다.

1.  **Docker 이미지 빌드:**
    ```bash
    docker build -t nlp-bert-api .
    ```

2.  **Docker 컨테이너 실행:**
    API는 `http://localhost:8000`에서 접근 가능합니다.
    ```bash
    docker run -d -p 8000:8000 nlp-bert-api
    ```

### API 엔드포인트

-   **`POST /predict/`**:
    -   JSON 형식 입력: `{"text": "여기에 문장을 입력하세요"}`
    -   JSON 형식 반환: `{"prediction": 0_또는_1, "text": "입력한 문장", "status": "success"}`
-   **`GET /health`**:
    -   JSON 형식 반환: `{"status": "ok"}`

## 기존 `nlp_practice.py` (레거시)

이 저장소는 이전에 간단한 Naive Bayes 분류기인 `nlp_practice.py`를 포함했습니다. 이 스크립트는 현재 더 고급 예제들의 맥락에서 레거시로 간주됩니다.

### 사용법 (레거시)
```bash
python3 nlp_practice.py "여기에 문장을 입력하세요"
```
명령줄에 문장이 제공되지 않으면 스크립트가 입력을 요청합니다.
