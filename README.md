# Lens dataset 관련 로직 👽

> NodeJS 20.10.0, NPM 10.2.3, PNPM 8.15.1

## (사용전) 미리 해야 하는 것들

1. 의존성 설치

```bash
pnpm install
```

2. 폴더 구조 확인

-   프로젝트 내 `./musinsa/brands.json` 파일이 준비되어 있어야 합니다.
-   스크래핑 결과를 저장할 `./output/` 폴더가 필요합니다. (코드 실행 시 자동 생성되지만, 권한 문제가 없는지 확인 필요)

## 디렉토리 구조

```bash
.
├── README.md                   # 프로젝트 개요 및 사용 방법
│
├── common.js                   # 로그/딜레이 등 공통 함수
├── collectProductInfo.js       # 브랜드별 상품 정보 수집
├── fetchReviews.js           # 상품 리뷰 데이터 수집
├── ...
│
├── musinsa/
│   └── brands.json               # 무신사 브랜드 목록 (예시)
│
└── output/
    ├── products/                 # 브랜드별 상품정보 JSON 저장
    └── review/                   # 리뷰 JSON 결과 저장
```

## 메인 로직

### 1. 브랜드 정보(상품정보) 저장

-   `./musinsa/brands.json` 파일 내 각 브랜드(`brandId`, `brandRepName`)에 대한 정보를 바탕으로 작업이 이루어집니다.
-   `collectProductInfo.js` 를 통해 해당 `brandId`에 대해 뮤신사 API를 호출하고, 상품 목록을 수집합니다.
    -   수집 결과는 `./output/products/{brandId}.json` 형태로 저장됩니다.

**예시** (`musinsa/brands.json`):

```json
[
    {
        "brandRepName": "에스피오나지",
        "brandId": "espionage",
        "products": []
    }
]
```

**상품 정보 결과 예시** (`./output/products/{brandId}.json`):

```json
{
  "brandRepName": "에스피오나지",
  "brandId": "espionage",
  "products": [
    {
      "images": ["https://image.msscdn.net/images/goods_img/..."],
      "brand": "espionage",
      "productId": "2568763",
      "link": "https://www.musinsa.com/app/goods/2568763",
      "name": "와이드 버뮤다 하프 슬랙스",
      "price": "37900",
      "salePrice": "37900",
      "soldOut": false
    },
    ...
  ]
}
```

**실행 방법**

```bash
node collectProductInfo.js
```

-   내부적으로 `./musinsa/brands.json` 을 읽고, 각 `brandId`별 상품 정보를 `./output/products/` 폴더에 저장합니다.

### 2. 브랜드별 상품 리뷰 데이터 추출

-   `fetchReviews.js` 에서 Puppeteer를 이용해 무신사 리뷰 페이지에 진입하고, 해당 페이지에서 호출되는 picture-reviews API를 가로채어 리뷰를 수집합니다.
-   모든 페이지(예: 1maxPage)를 순회하며 0.51초 사이의 랜덤 딜레이를 적용합니다.
-   결과는 `./output/review/{brandId}/{productId}.json` 에 저장됩니다.

**예시**:

```json
{
  "brandId": "espionage",
  "productId": "2568763",
  "totalCount": 1094,
  "reviews": [
    {
      "reviewNo": 123456,
      "goodsNo": 2568763,
      "contents": "옷이 편하고 좋아요!",
      "images": [
        {
          "image": "https://image.msscdn.net/thumbnails/..."
        }
      ],
      ...
    },
    ...
  ]
}
```

**실행 방법**

```bash
node fetchReviews.js <brandId>
```

-   내부적으로 `./output/products/{brandId}.json`의 `products` 배열을 순회하며, 각 `productId`별 리뷰를 수집합니다.
