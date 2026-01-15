# ML4T (Machine Learning for Trading) Project

## Project Overview

**Stefan Jansen**의 "Machine Learning for Trading" 2판 교재 공식 GitHub 레포지토리 (fork)
- **원본**: https://github.com/stefan-jansen/machine-learning-for-trading
- **책**: Amazon에서 구매 가능 (2nd Edition, 2020)
- **커뮤니티**: https://exchange.ml4trading.io/

---

## Repository Structure

### Chapter Organization (23 Chapters + Appendix)

```
00_machine_learning_for_trading/     # Ch 1: ML for Trading 소개
01_machine_learning_for_trading/
02_market_and_fundamental_data/      # Ch 2: 시장 데이터 & 펀더멘털
03_alternative_data/                 # Ch 3: 대안 데이터
04_alpha_factor_research/            # Ch 4: 알파 팩터 연구
05_strategy_evaluation/              # Ch 5: 전략 평가
06_machine_learning_process/         # Ch 6: ML 프로세스
07_linear_models/                    # Ch 7: 선형 모델
08_ml4t_workflow/                    # Ch 8: ML4T 워크플로우 (Zipline)
09_time_series_models/               # Ch 9: 시계열 모델 (ARIMA, GARCH)
10_bayesian_machine_learning/        # Ch 10: 베이지안 ML
11_decision_trees_random_forests/    # Ch 11: 의사결정 트리 & 랜덤 포레스트
12_gradient_boosting_machines/       # Ch 12: GBM (XGBoost, LightGBM)
13_unsupervised_learning/            # Ch 13: 비지도학습
14_working_with_text_data/           # Ch 14: NLP & 감성분석
15_topic_modeling/                   # Ch 15: 토픽 모델링 (LDA)
16_word_embeddings/                  # Ch 16: Word2Vec, BERT
17_deep_learning/                    # Ch 17: 딥러닝 기초
18_convolutional_neural_nets/        # Ch 18: CNN
19_recurrent_neural_nets/            # Ch 19: RNN/LSTM
20_autoencoders_for_conditional_risk_factors/  # Ch 20: Autoencoder
21_gans_for_synthetic_time_series/   # Ch 21: GAN
22_deep_reinforcement_learning/      # Ch 22: 강화학습
23_next_steps/                       # Ch 23: 마무리
24_alpha_factor_library/             # Appendix: 알파 팩터 라이브러리
data/                                # 데이터셋 (~3.2GB)
installation/                        # 설치 가이드
```

---

## Part Summary

| Part | Chapters | Topics |
|------|----------|--------|
| **Part 1: Data** | Ch 2-3 | 시장 데이터, 펀더멘털, 대안 데이터 (NASDAQ ITCH, SEC EDGAR) |
| **Part 2: Supervised Learning** | Ch 4-12 | 피처 엔지니어링, 선형 모델, 트리 모델, GBM, 시계열, 베이지안 |
| **Part 3: NLP** | Ch 14-16 | 텍스트 분석, 토픽 모델링, 워드 임베딩 |
| **Part 4: Deep Learning** | Ch 17-22 | CNN, RNN, Autoencoder, GAN, 강화학습 |

---

## Tech Stack

### Core ML/Data Science
- NumPy, SciPy, pandas, scikit-learn, statsmodels

### Deep Learning
- TensorFlow 2.x, Keras, PyTorch

### Trading & Backtesting
- Zipline-reloaded, PyFolio-reloaded, Alphalens-reloaded, BackTrader

### Gradient Boosting
- XGBoost, LightGBM, CatBoost

### NLP
- spaCy, Gensim, TextBlob, PyLDAvis

### Specialized
- TA-Lib (기술적 분석)
- PyKalman (칼만 필터)
- PyWavelets (웨이블릿)
- ARCH (변동성 모델링)
- PyMC3 (베이지안 추론)

---

## Data Sources (~3.2GB)

| Dataset | Size | Description |
|---------|------|-------------|
| Wiki prices | 1.7GB | 과거 주가 데이터 |
| assets.h5 | 1.5GB | 전처리된 시장 데이터 (HDF5) |
| STOOQ | - | 일본 주식, US ETF |
| SEC EDGAR | - | 기업 공시 (XBRL) |
| Earnings calls | 9.9MB | 실적 발표 콜 |
| GloVe vectors | - | 사전학습 워드 임베딩 |

---

## ML4T Workflow

```
1. Generate ideas (아이디어 생성)
   ↓
2. Collect data (데이터 수집)
   ↓
3. Feature engineering (피처 엔지니어링)
   ↓
4. Model design & tuning (모델 설계 & 튜닝)
   ↓
5. Strategy design (전략 설계)
   ↓
6. Backtesting (백테스팅)
   ↓
7. Performance evaluation (성과 평가)
   ↓
   (iterate)
```

---

## Learning Progress

- [x] Ch 1: Machine Learning for Trading
- [ ] Ch 2: Market and Fundamental Data
- [ ] Ch 3: Alternative Data
- [ ] Ch 4: Alpha Factor Research (진행 중)
- [ ] Ch 5-23: 대기 중

---

## Notes

### Ch 4: Alpha Factor Research
- 피처 엔지니어링 기법: TA-Lib, Kalman Filter, Wavelets
- 결측값 처리: 티커별 그룹으로 평균 대체
- 노트북: 01_feature_engineering.ipynb ~ 05_*.ipynb

---

## Environment Setup

설치 가이드: `installation/README.md` 참조

```bash
# Conda 환경 생성 (Windows)
conda env create -f installation/windows/ml4t.yml
conda activate ml4t
```
