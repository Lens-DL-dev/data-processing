## data-processing

```bash
.
├── README.md               # 프로젝트 개요 (현재 문서)
├── scrap/                  # 데이터 수집 로직
│   ├── README.md           # scrap 사용법
│   ├── collectProductInfo.js
│   ├── fetchReviews.js
│   ├── ...
│   ├── musinsa/
│   │   └── brands.json
│   └── output/
│       ├── products/
│       └── review/
├── preprocessing/          # 데이터 정제 파이프라인
│   ├── README.md           # preprocessing 사용법
│   ├── refine.py
│   ├── requirements.txt
│   ├── ...
│   ├── dataset/
│   │   ├── product/
│   │   └── wearing/
│   └── output/
│       ├── results/
│       ├── segmentation_results/
│       ├── cropped/
│       └── ...
```