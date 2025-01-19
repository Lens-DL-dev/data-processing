// fetchReviews.js
import fs from 'fs';
import path from 'path';
import puppeteer from 'puppeteer';
import { logInfo, logError, delay, randomShortDelay, saveLog } from './common.js';

/**
 * 단일 상품(productId)에 대한 리뷰를 모두 수집하여 json 저장
 * @param {puppeteer.Browser} browser
 * @param {string} brandId
 * @param {string} productId
 * @returns {Promise<void>}
 */
async function fetchReviewsForProduct(browser, brandId, productId) {
    const functionName = 'fetchReviewsForProduct';
    logInfo(functionName, `Start fetching reviews for productId=${productId}`);

    // 저장 폴더 준비
    const dirPath = path.join('./output/review', brandId);
    fs.mkdirSync(dirPath, { recursive: true });
    const outputPath = path.join(dirPath, `${productId}.json`);

    // 수집한 리뷰 전체 배열
    const allReviews = [];

    // 새 페이지 열기
    const page = await browser.newPage();
    await page.setUserAgent(
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99 Safari/537.36'
    );

    // 이 상품에 대해 "picture-reviews" 엔드포인트로 들어오는 응답을 캐치하기 위한 핸들러
    // pageNum 별로 요청을 발생시킬테니, 그때마다 응답 JSON 파싱
    page.on('response', async response => {
        try {
            const url = response.url();
            if (url.includes('picture-reviews')) {
                // JSON 파싱
                const json = await response.json();
                const reviews = json?.data?.list ?? [];
                // 이미지 경로 치환
                for (const rev of reviews) {
                    // images 배열의 image 경로 치환
                    if (Array.isArray(rev.images)) {
                        rev.images.forEach(img => {
                            if (img.image && !img.image.startsWith('http')) {
                                img.image = `https://image.msscdn.net${img.image}`;
                            }
                        });
                    }
                }
                // 수집한 리뷰들을 하나의 배열에 합침
                allReviews.push(...reviews);
            }
        } catch (err) {
            logError(functionName, `Error parsing review response: ${err.message}`);
        }
    });

    try {
        // 1) 리뷰 페이지 이동
        const reviewUrl = `https://www.musinsa.com/review/goods/${productId}`;
        await page.goto(reviewUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });

        // 2) 첫 페이지 리뷰 API 호출 -> totalCount 확인
        const firstApi = `https://goods.musinsa.com/api2/review/v1/picture-reviews?goodsNo=${productId}&size=50&page=1`;
        // 페이지 내부에서 fetch 호출
        await page.evaluate(apiUrl => {
            return fetch(apiUrl).then(res => res.json());
        }, firstApi);

        // 약간 대기 (응답 수신까지)
        await delay(2000);

        // totalCount를 찾아서 pageNum 계산 (이미 on('response') 에서 firstApi 응답이 들어왔으므로, on('response') 없이 직접 가져오는 방법)
        // 직접 page.evaluate 로 첫 API 응답 다시 수집하는 것도 가능
        // 여기서는 단순히 allReviews 길이가 1페이지치 수집된 후, totalCount만 추가로 가져오는 식(예: 다시 evaluate)
        let totalCount = 0;
        const firstRespData = await page.evaluate(apiUrl => {
            return fetch(apiUrl).then(res => res.json());
        }, firstApi);
        totalCount = firstRespData?.data?.totalCount ?? 0;

        const maxPage = Math.ceil(totalCount / 50);
        logInfo(functionName, `totalCount=${totalCount}, maxPage=${maxPage}`);

        // 3) 나머지 페이지 순회
        for (let pageNum = 2; pageNum <= Math.min(maxPage, 3); pageNum++) {
            const apiUrl = `https://goods.musinsa.com/api2/review/v1/picture-reviews?goodsNo=${productId}&size=50&page=${pageNum}`;
            logInfo(functionName, `Fetch page=${pageNum}/${maxPage}`);
            await page.evaluate(url => {
                return fetch(url).then(res => res.json());
            }, apiUrl);

            // 0.5~1초 랜덤 딜레이
            await randomShortDelay();
        }

        // 모든 요청의 응답을 받기 위해 잠시 대기
        await delay(2000);

        // 4) 최종 리뷰 데이터 저장
        const outputJson = {
            brandId,
            productId,
            totalCount,
            reviews: allReviews,
        };
        fs.writeFileSync(outputPath, JSON.stringify(outputJson, null, 2), 'utf-8');
        logInfo(functionName, `Saved reviews => ${outputPath} (reviews.length=${allReviews.length})`);
    } catch (err) {
        logError(functionName, `Error on fetchReviewsForProduct(${productId}): ${err.message}`);
    } finally {
        await page.close();
    }
}

/**
 * 브랜드 ID에 해당하는 products 데이터를 읽은 뒤,
 * 각 productId에 대해 fetchReviewsForProduct 실행
 * @param {string} brandId
 */
export async function fetchReviewsForBrand(brandId) {
    const functionName = 'fetchReviewsForBrand';
    // 1) 브랜드 상품정보 로드
    const productFilePath = `./output/products/${brandId}.json`;
    if (!fs.existsSync(productFilePath)) {
        logError(functionName, `Product file not found: ${productFilePath}`);
        return;
    }

    const raw = fs.readFileSync(productFilePath, 'utf-8');
    const brandData = JSON.parse(raw);
    const products = brandData.products || [];
    if (!products.length) {
        logInfo(functionName, `No products found for brandId="${brandId}"`);
        return;
    }

    // 2) Puppeteer 브라우저 준비
    const browser = await puppeteer.launch({
        headless: true,
    });

    try {
        // 3) 각 productId 순회하며 리뷰 수집
        for (let i = 0; i < products.length; i++) {
            const product = products[i];
            const productId = product.productId;
            if (!productId) continue; // 안전 처리

            await fetchReviewsForProduct(browser, brandId, productId);
            // 너무 빠른 요청 방지용 (랜덤 딜레이)
            await randomShortDelay();
        }
    } catch (error) {
        logError(functionName, `Error in fetchReviewsForBrand: ${error.message}`);
    } finally {
        await browser.close();
    }
}

/**
 * 이 파일을 단독 실행하면, 예시로 특정 brandId 에 대해 리뷰 수집을 진행
 */

(async () => {
    const targetBrandId = process.argv[2]; // 예: node fetchReviews.js musinsastandard
    if (!targetBrandId) {
        console.error('Usage: node fetchReviews.js <brandId>');
        process.exit(1);
    }
    await fetchReviewsForBrand(targetBrandId);
})();
