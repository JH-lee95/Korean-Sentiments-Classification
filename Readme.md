# Introduction
-  한국어 BERT 모델을 이용한 감성분류 모델 (DistilKoBERT 추가, 210811)
- 대화데이터셋을 이용한, 대화체에서의 감성분류
- 7가지 감정에 대한 Multi-category 분류

# Preparing Datasets

## Datasets
1. 감성대화 말뭉치   
(https://aihub.or.kr/aidata/7978)

감성대화 말뭉치 데이터셋을 다운로드 받으면 다음과 같은 형태의 자료가 준비되어 있습니다.  
저희는 [Training]과 [Validation]의 원천데이터와 최종데이터에 들어있는 xlsx파일을 사용합니다.

```
|- [감성대화]/
|   |- Training/
        |- 감성대화말뭉치(원천데이터)_Training.zip
            |- 감성대화말뭉치(원시데이터)_Training.xlsx
            |- 감성대화말뭉치(원시데이터)_Training.json
        |- 감성대화말뭉치(최종데이터)_Training.zip
            |- ...
|   |- Validation/
        |- ...
|   |- 원천데이터/
|       |- ...
```

2. 한국어 감정 정보가 포함된 연속적 대화 데이터셋  
(https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-010)


3. 한국어 감정 정보가 포함된 단발성 대화 데이터셋  
(https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-009)

## Preprocess Datasets

위의 두 데이터셋을 준비하였으면, 전처리 작업을 진행합니다.  

전처리는 [dataset_preprocessing.ipynb](./dataset_preprocessing.ipynb)을 실행시킵니다.  
노트북 스크립트를 모두 실행하면 [sentiment_dialogues.csv]라는 파일이 생기며, 이 파일이 학습을 위한 최종 데이터셋이 됩니다.


# Training and Inference

[run_KoBERT_classifier.ipynb](./run_KoBERT_classifier.ipynb)의 스크립트를 실행시키면, 학습부터 Inference까지 모든 동작이 가능합니다. 이때 학습에 사용되는 데이터는 위의 과정을 거치면 생성된 [sentiment_dialogues.csv]입니다.


# References

- [감성대화 말뭉치](https://aihub.or.kr/aidata/7978)

- [한국어 감정 정보가 포함된 연속적 대화 데이터셋](https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-010)

- [한국어 감정 정보가 포함된 단발성 대화 데이터셋](https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-009)

- [SKTBrain KoBERT](https://github.com/SKTBrain/KoBERT)

- [koo's tech diary](https://tech-diary.tistory.com/31)

- [DistillKoBERT](https://github.com/monologg/DistilKoBERT)

- [SNNLP](http://knlp.snu.ac.kr/)
