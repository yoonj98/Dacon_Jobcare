# Dacon 잡케어 추천 알고리즘 경진대회
#### 팀 훠궈 / 🥈우수상 수상🥈
---
      한국고용정보원에서 제공하는 구인구직 빅데이터 기반 커리어 관리 서비스인 
      잡케어 데이터를 통해 개인별 맞춤형 컨텐츠 추천 모델 구축 및 활용 방안을 제시한 프로젝트

## 📌 전체 실행 프로세스
#### 1. Catboost / Optuna 설치
#### 2. 라이브러리 불러오기
  - 개발 환경 및 라이브러리 버전 확인    

#### 3. 데이터 불러오기
#### 4. 탐색적 자료분석 (EDA)
  - 데이터 기초 통계량 확인
  - 데이터 결측치 및 불균형 확인
  - 데이터 시각화

#### 5. 데이터 전처리 
  - Boolean형 변수 label encoding
  - 파생변수 생성
    - 컨텐츠 열람 일시 변수 → 요일과 시간 관련 변수 생성 (contents_open_wd, contents_open_hour, contents_weekday, contents_work_time)
    - 컨텐츠 번호 빈도수 변수 생성 (contents_rn_cnt)
    - 사용자 번호 빈도수 변수 생성 (person_rn_cnt)
    - 속성 D의 대분류 매칭 여부 변수 생성 (d_1_l_match_yn, d_2_l_match_yn, d_3_l_match_yn)
    - 속성 D의 코드 매칭 여부 변수 생성 (d_1_s_match_yn, d_2_s_match_yn, d_3_s_match_yn)
  - 변수 삭제 → label이 하나거나, 파생변수를 생성하는 데 사용한 일부 변수 제거 

  >    id, person_prefer_f ,person_prefer_g, person_rn, contents_rn, contents_open_dt, d_l_match_yn, d_m_match_yn, d_s_match_yn, h_m_match_yn, h_s_match_yn, person_prefer_d_1_l, person_prefer_d_2_l,person_prefer_d_3_l, contents_attribute_d_l

#### 6. 모델링
  - 데이터에 범주형 변수의 비중이 높기 때문에 Catboost 모델을 사용
  - Optuna 라이브러리를 통해 최적의 하이퍼 파리미터 탐색 (F1 score maximize, Trial 10)
  - K-fold 교차 검증 진행 (n_splits = 5)
  - CV별 예측 확률을 평균 내어 최종 예측 확률로 활용
  - threshold = 0.4를 기준으로 예측 확률을 label로 변환

<br>

## 📌 Presentation
저희 프로젝트에 대해 자세하게 알고 싶으시다면, 프로젝트 설명자료를 참고해주세요. 
* [![GoogleDrive Badge](https://img.shields.io/badge/Presentation-405263?style=flat-square&logo=Quip&link=https://drive.google.com/file/d/1wkLDchFS6nExMgtldQYKGfVSP6YOjCl-/view?usp=sharing)](https://drive.google.com/file/d/1wkLDchFS6nExMgtldQYKGfVSP6YOjCl-/view?usp=sharing)

<br>

## 📌 Structure
```python
훠궈  
├── README.md
├── Final_Code.ipynb
├── data  
│    ├───train.csv
│    ├───test.csv
│    ├───result_submission.csv
│    ├───train_data.csv
│    └───test_data.csv
│          
├── preprocess
│    ├───EDA.ipynb
│    └───preprocess.ipynb
│    
└── model
     ├───hyper_parameter.ipynb
     ├───model.ipynb
     └─── model
           └───catboost_optuna_parameter.pkl
```
<br>

### 📌 개발 환경 및 라이브러리 버전
```python
OS                            Linux-5.4.0-91-generic-x86_64-with-debian-buster-sid
Process information           x86_64
Process Architecture          x86_64
RAM                           252 GB

python                        3.7.6
numpy                         1.18.1
pandas                        1.0.1
scikit-learn                  0.22.1
catboost                      1.0.4
optuna                        2.10.0
```

<br>

## 📌 Contributors
<table>
  <tr>
    <td align="center"><a href="https://github.com/yoonj98"><b>이윤정</b></sub></td>
    <td align="center"><a href="https://github.com/jiwon4178"><b>박지원</b></sub></td>
    <td align="center"><a href="https://github.com/jihyeon4028"><b>박지현</b></sub></td>
    <td align="center"><a href="https://github.com/didwldn3032"><b>양지우</b></sub></td>
</table>

