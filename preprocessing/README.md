# Lens-DL Dataset 정제 파이프라인

## 설치

### 1. 가상 환경 생성

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows
```

### 2. 필수 패키지 설치

필요한 Python 라이브러리를 설치

```bash
pip install -r requirements.txt
```

### 3. CUDA 가용성 확인

병렬처리를 위해 GPU 가속이 가능한지 체크하기

```python
import torch
print("CUDA Available:", torch.cuda.is_available())
```

## 데이터셋

### 데이터셋 다운로드

```
<dataset_dir>/
  ├── product/      # 상품 이미지 (*.jpg)
  └── wearing/      # 착용 이미지 (*.jpg)
```

데이터셋을 `dataset` 디렉토리에 배치하거나, `--dataset_dir` 인수를 사용하여 경로를 지정

## 인수 설명

파이프라인은 다음 인수를 받습니다:

-   `--product_id` : 처리할 단일 상품 ID 지정 (예: `2792167`).
-   `--dataset_dir`: `product` 및 `wearing` 폴더를 포함하는 데이터셋 디렉토리 경로. 기본값: `dataset`.
-   `--output_dir` : 결과를 저장할 루트 디렉토리. 기본값: `output`.
-   `--worker` : 병렬 세그멘테이션에 사용할 프로세스 수. 기본값: `1`.

## 사용법

### 단일 상품 처리 예시

특정 상품 ID를 처리:

```bash
python refine.py --product_id 2792167 --dataset_dir dataset --output_dir output --worker 8
```

### 전체 데이터셋 처리 예시

데이터셋 내 모든 상품 ID를 처리:

```bash
python refine.py --dataset_dir dataset --output_dir output --worker 4
```

## 로직 상세

### 1. YOLO 검출

YOLO 모델을 사용하여 `wearing` 이미지에서 의류를 검출합니다. 결과는 임계값에 따라 Valid 혹은 Invalid 처리

-   **출력 디렉토리**: `output/results/<product_id>/`
-   **생성 파일**:
    -   `results.json`: Valid 혹은 Invalid 분류 결과.
    -   `results_detail.json`: 세부 검출 데이터.

### 2. 검증 및 크롭

Segformer 모델을 사용하여 `wearing` 이미지를 검증하고 `product` 이미지와 색상을 비교합니다. 유효한 검출에 대해 크롭 이미지를 생성

-   **출력 디렉토리**:
    -   `output/segmentation_results/<product_id>/`
    -   `output/cropped/<product_id>/`
-   **생성 파일**:
    -   `validation_results.json`: 세그멘테이션 이후 유효/무효 분류 결과.
    -   `summary.json`: 세그멘테이션 분석 데이터.

### 3. 정리

결과를 집계하고 요약 시각화를 생성합니다.

-   **출력 디렉토리**:
    -   `output/wearing/`: 크롭된 이미지.
    -   `output/product/`: 상품 이미지.
    -   `output/summary_<product_id>.png`: 요약 시각화.
