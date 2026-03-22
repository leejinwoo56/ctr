 # click-prediction-model

정형 피처와 시퀀스 피처를 함께 활용하여 클릭 여부를 예측하는 딥러닝 모델을 구현한 프로젝트입니다.  
PyTorch를 사용해 tabular input과 sequence input을 결합한 hybrid architecture를 구성했으며, 시퀀스 정보는 2-layer LSTM과 attention pooling으로 인코딩하고, 최종적으로 binary classification을 통해 클릭 확률을 예측합니다.  
이 프로젝트를 통해 실제 예측 파이프라인에서 데이터 로딩, 불균형 데이터 샘플링, sequence modeling, ROC-AUC 기반 검증, early stopping, mixed precision inference 및 training 기법을 구현 중심으로 익혔습니다.
## Project

### Click Prediction with Tabular + Sequence Modeling

**File:** `src/click_prediction_lstm_attention.py`

#### 개요
정형 feature와 sequence feature를 함께 입력으로 받아 클릭 여부를 예측하는 binary classification 프로젝트입니다.  
정형 데이터는 정규화 후 MLP에 입력하고, sequence 데이터는 LSTM과 attention pooling을 통해 표현 벡터로 변환한 뒤 결합하여 최종 클릭 확률을 예측합니다.

#### 구현 내용
- `train.parquet`, `test.parquet` 기반 데이터 로딩
- 클릭/비클릭 데이터 비율을 조정한 resampling 적용
- tabular feature와 sequence feature를 함께 처리하는 custom `Dataset` 구현
- variable-length sequence를 위한 padding 및 `collate_fn` 구현
- 2-layer LSTM 기반 sequence encoder 구성
- attention pooling을 통해 sequence representation 추출
- tabular feature와 sequence representation을 결합한 MLP classifier 구현
- `BCEWithLogitsLoss` 기반 binary classification 학습
- validation ROC-AUC 기반 모델 선택 및 early stopping 적용
- mixed precision training, gradient clipping, cosine annealing scheduler 사용
- test inference 후 submission 파일 생성

#### 배운 점
- tabular 데이터와 sequential 데이터를 함께 사용하는 hybrid model 구조를 설계하는 방법
- variable-length sequence를 LSTM에 효율적으로 입력하기 위한 padding / packing 처리 방식
- attention pooling이 sequence 전체 hidden state에서 중요한 정보를 선택하는 방식
- 불균형 binary classification 문제에서 ROC-AUC를 활용해 모델을 평가하는 방법
- 실제 학습 파이프라인에서 early stopping, mixed precision, scheduler, gradient clipping 같은 안정화 기법의 역할

---

## Skills Demonstrated

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Binary Classification
- Click Prediction
- LSTM
- Attention Pooling
- Tabular + Sequential Feature Modeling
- Custom Dataset / DataLoader
- ROC-AUC Evaluation
- Mixed Precision Training
- Early Stopping
- Gradient Clipping
- Learning Rate Scheduling
