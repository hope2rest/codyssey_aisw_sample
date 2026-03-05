# codyssey_aisw_sample

AI/SW 심화 시험 — pytest 기반 자동 채점 멀티 모듈 프로젝트

## 문제 개요

### Q1. 이미지 기반 객체 카운팅 (CV)

제공되는 이미지(easy/medium/hard 각 5장)에서 규칙 기반 파이프라인으로 박스 개수를 카운팅하고, 데이터 증강 앙상블로 정확도를 개선한 뒤, 객체 검출 결과를 일주일 단위로 관리하는 일지 시스템을 구축하는 문제입니다.

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
    │       └── test_q1_cv.py       # 22개 테스트
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
| `conv2d.py` | NumPy 2D 컨볼루션, 그레이스케일 변환, Sobel 엣지 검출, 이미지 증강(좌우/상하 반전, 밝기 조절, 정규화) |
| `counter.py` | 기본 박스 카운팅, 증강 앙상블 카운팅, 바운딩 박스 좌표 추출 |
| `metrics.py` | MAE/Accuracy 계산, worst case 탐색, 기본 vs 증강 비교, 일자별 검출 로그, 주간 보고서, 한계 분석 |
| `main.py` | 전체 파이프라인 실행 (카운팅 → 증강 → 비교 → 주간 일지 → JSON 저장) |
| `result_q1.json` | 예측, 증강 예측, 바운딩 박스, 메트릭, 방법 비교, 주간 보고서, 한계 분석 |

### Intro — MAC 스코어러

| 파일 | 역할 |
|------|------|
| `mac_scorer.py` | 음악 파일 정보를 파싱하여 MAC 점수를 계산하는 모듈 |

## 테스트 구성

### CV (22개)

| 분류 | 테스트 | 내용 |
|------|--------|------|
| 구조 검증 | `test_conv2d_functions` | conv2d.py 필수 함수 7개 정의 (AST) |
| 구조 검증 | `test_no_filter2d` | cv2.filter2D 미사용 확인 |
| 구조 검증 | `test_counter_functions` | counter.py 필수 함수 4개 + THRESHOLD/MIN_AREA |
| 구조 검증 | `test_metrics_functions` | metrics.py 필수 함수 7개 정의 (AST) |
| 기능 검증 | `test_conv2d_identity` | identity 커널 conv2d 결과 |
| 기능 검증 | `test_conv2d_sobel` | Sobel 커널 적용 결과 |
| 기능 검증 | `test_grayscale` | 그레이스케일 변환 정확도 |
| 기능 검증 | `test_flip_operations` | 좌우/상하 반전 검증 |
| 기능 검증 | `test_brightness_and_normalize` | 밝기 조절 및 Min-Max 정규화 |
| 기능 검증 | `test_ensemble_count` | 중앙값 기반 앙상블 카운팅 |
| 기능 검증 | `test_count_boxes_augmented` | 증강 앙상블 카운팅 정수 반환 |
| 기능 검증 | `test_bounding_boxes_format` | 바운딩 박스 출력 형식 |
| 기능 검증 | `test_compute_metrics` | MAE/Accuracy 계산 정확도 |
| 기능 검증 | `test_find_worst_case` | worst case 이미지 탐색 |
| 기능 검증 | `test_compare_methods` | 기본 vs 증강 성능 비교 |
| 기능 검증 | `test_detection_log_and_weekly_report` | 검출 로그 및 주간 보고서 |
| 결과 검증 | `test_valid_images_only` | 유효 이미지만 분석 (test_01 제외) |
| 결과 검증 | `test_augmented_predictions` | 증강 카운팅 결과 포함 |
| 결과 검증 | `test_method_comparison` | 방법 비교 결과 포함 |
| 결과 검증 | `test_weekly_report` | 주간 보고서 결과 포함 |
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
