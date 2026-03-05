# codyssey_aisw_sample

AI/SW 심화 시험 — pytest 기반 자동 채점 멀티 모듈 프로젝트

## 문제 개요

### Q1. 이미지 기반 객체 카운팅 (CV)

제공되는 이미지(easy/medium/hard 각 5장)에서 규칙 기반 파이프라인으로 박스 개수를 카운팅하고, 카테고리별 성능 지표를 산출한 뒤 규칙 기반 방식의 한계를 분석하는 문제입니다.

학생은 4개의 Python 모듈과 결과 JSON 파일, 총 5개 파일을 제출합니다.

### Q1. MAC 스코어러 (Intro)

음악 콘텐츠 파일 정보를 파싱하여 MAC(Music Audio Content) 점수를 계산하는 문제입니다.

학생은 `mac_scorer.py` 1개 파일을 제출합니다.

## 디렉토리 구조

```
codyssey_aisw_sample/
├── conftest.py                     # 루트: --submission-dir CLI 옵션 등록
├── pyproject.toml                  # pytest 설정
├── requirements.txt                # pytest>=7.0
└── missions/
    ├── cv/level2/mission01/        # Q1 이미지 객체 카운팅
    │   ├── problem.md              # 문제지 (학생 배포용)
    │   ├── solution.md             # 정답지 (출제자 보관용)
    │   ├── data/
    │   │   ├── labels.json         # 이미지별 정답 박스 개수
    │   │   └── images/             # easy/medium/hard 각 5장 (15장)
    │   ├── template/               # 학생 배포용 스켈레톤 코드
    │   │   ├── conv2d.py
    │   │   ├── counter.py
    │   │   ├── metrics.py
    │   │   └── main.py
    │   ├── sample_submission/      # 정답 예시 코드
    │   │   ├── conv2d.py
    │   │   ├── counter.py
    │   │   ├── metrics.py
    │   │   ├── main.py
    │   │   └── result_q1.json
    │   └── tests/
    │       ├── conftest.py         # submission_dir fixture
    │       └── test_q1_cv.py       # 11개 테스트
    └── intro/level1/mission01/     # Q1 MAC 스코어러
        ├── problem.md              # 문제지 (학생 배포용)
        ├── solution.md             # 정답지 (출제자 보관용)
        ├── data/
        │   └── data.json           # 음악 파일 정보
        ├── template/               # 학생 배포용 스켈레톤 코드
        │   └── mac_scorer.py
        ├── sample_submission/      # 정답 예시 코드
        │   └── mac_scorer.py
        └── tests/
            ├── conftest.py         # submission_dir fixture
            └── test_mac_scorer.py  # 9개 테스트
```

## 제출 파일 설명

### CV — 이미지 기반 객체 카운팅

| 파일 | 역할 |
|------|------|
| `conv2d.py` | NumPy만으로 2D 컨볼루션을 구현하고, 그레이스케일 변환 및 Sobel 엣지 검출을 수행 |
| `counter.py` | `conv2d.py`를 import하여 이미지 로드 → 엣지 검출 → 이진화 → Connected Component 분석 → 최소 면적 필터로 박스 개수를 카운팅 |
| `metrics.py` | 카테고리별 MAE/Accuracy 계산, worst case 탐색, 규칙 기반 실패 원인 및 학습 기반 접근법 필요성 서술 |
| `main.py` | `counter.py`와 `metrics.py`를 import하여 전체 파이프라인을 실행하고 `result_q1.json`을 생성 |
| `result_q1.json` | 예측 결과, 카테고리별 메트릭, worst case, 실패 원인, 학습 기반 필요성을 담은 최종 출력 |

### Intro — MAC 스코어러

| 파일 | 역할 |
|------|------|
| `mac_scorer.py` | 음악 파일 정보를 파싱하여 MAC 점수를 계산하는 모듈 |

## 테스트 구성

### CV (11개)

| 분류 | 테스트 | 내용 |
|------|--------|------|
| 구조 검증 | `test_conv2d_functions` | conv2d.py 필수 함수 3개 정의 (AST) |
| 구조 검증 | `test_no_filter2d` | cv2.filter2D 미사용 확인 |
| 기능 검증 | `test_conv2d_identity` | identity 커널 conv2d 결과 |
| 기능 검증 | `test_conv2d_sobel` | Sobel 커널 적용 결과 |
| 기능 검증 | `test_grayscale` | 그레이스케일 변환 정확도 |
| 기능 검증 | `test_counter_functions` | count_boxes, THRESHOLD, MIN_AREA 정의 확인 |
| 기능 검증 | `test_compute_metrics` | MAE/Accuracy 계산 정확도 |
| 기능 검증 | `test_find_worst_case` | worst case 이미지 탐색 |
| 결과 검증 | `test_valid_images_only` | 유효 이미지만 분석 (test_01 제외) |
| 결과 검증 | `test_failure_reasons` | 실패 원인 3개 이상, 한국어, 20자+ |
| 결과 검증 | `test_why_learning_based` | 학습 기반 필요성 30~200자, 한국어 |

### Intro (9개)

| 분류 | 테스트 | 내용 |
|------|--------|------|
| 구조 검증 | `test_class_exists` | MACScorer 클래스 존재 확인 |
| 구조 검증 | `test_methods_exist` | 필수 메서드 존재 확인 |
| 기능 검증 | `test_load_data` | 데이터 로드 기능 |
| 기능 검증 | `test_parse_duration` | 시간 파싱 기능 |
| 기능 검증 | `test_calculate_score` | 점수 계산 기능 |
| 기능 검증 | `test_get_top_items` | 상위 항목 반환 |
| 기능 검증 | `test_get_statistics` | 통계 계산 |
| 기능 검증 | `test_filter_by_category` | 카테고리 필터링 |
| 결과 검증 | `test_export_results` | 결과 JSON 출력 |

## 실행 방법

```bash
# 정답 코드로 검증 (기본값: sample_submission)
pytest missions/cv/level2/mission01/tests/ -v
pytest missions/intro/level1/mission01/tests/ -v

# 학생 제출물 채점
pytest missions/cv/level2/mission01/tests/ --submission-dir /path/to/submission -v
pytest missions/intro/level1/mission01/tests/ --submission-dir /path/to/submission -v

# zip 파일로 제출 시
pytest missions/cv/level2/mission01/tests/ --submission-dir /path/to/submission.zip -v
```
