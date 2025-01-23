// collectProductInfo.js
import fs from 'fs';
import axios from 'axios';
import { logInfo, logError, delay, saveLog } from './common.js';

/**
 * 뮤신사 API에서 브랜드별 상품 리스트를 수집하고,
 * convert.py 와 동일한 형태의 product 데이터로 변환한 후 반환합니다.
 *
 * @param {string} brandId    - 브랜드 ID (예: "musinsastandard")
 * @param {string} brandRepName - 브랜드 대표명 (예: "무신사스탠다드")
 * @returns {Promise<Object>} - { brandRepName, brandId, products: [...] } 형태의 객체
 */
export async function collectProductInfo(brandId, brandRepName) {
    const functionName = 'collectProductInfo';
    logInfo(functionName, `Start collecting products for brandId="${brandId}"`);

    // 1) 첫 페이지 호출 -> totalCount(또는 total) 파악
    let totalCount = 0;
    let totalPages = 0;
    const sizePerPage = 100; // API에서 한 번에 가져오는 상품 수

    const firstUrl = `https://api.musinsa.com/api2/dp/v1/plp/goods?brand=${brandId}&gf=A&sortCode=POPULAR&page=1&size=${sizePerPage}&caller=BRAND`;
    try {
        const firstRes = await axios.get(firstUrl, { headers: getRandomUserAgentHeader() });

        const firstData = firstRes.data;

        // pagination 객체에서 직접 값을 가져옵니다
        totalCount = firstData?.data?.pagination?.totalCount ?? 0;
        totalPages = firstData?.data?.pagination?.totalPages ?? 0;

        logInfo(functionName, `Detected totalCount=${totalCount}, totalPages=${totalPages}`);
    } catch (err) {
        logError(functionName, `Failed to fetch first page for brandId="${brandId}": ${err.message}`);
        return { brandRepName, brandId, products: [] };
    }

    // 2) 모든 페이지 순회하여 상품정보 수집
    const allProducts = [];
    for (let page = 1; page <= totalPages; page++) {
        const url = `https://api.musinsa.com/api2/dp/v1/plp/goods?brand=${brandId}&gf=A&sortCode=POPULAR&page=${page}&size=${sizePerPage}&caller=BRAND`;
        try {
            const response = await axios.get(url, { headers: getRandomUserAgentHeader() });
            const data = response.data?.data;

            // products array 추출 (convert.py에서 data['list'] 참고)
            const products_a = data?.list ?? [];

            // convert.py 로직과 유사하게 변환
            for (const product_a of products_a) {
                const saleRate = product_a?.saleRate ?? 0;
                // 판매가/정상가 처리
                const productPrice = saleRate > 0 ? product_a?.normalPrice : product_a?.price;

                const product_b = {
                    images: [product_a?.thumbnail ?? ''],
                    brand: brandId,
                    link: product_a?.goodsLinkUrl ?? '',
                    productId: product_a?.goodsNo?.toString() ?? '',
                    name: product_a?.goodsName ?? '',
                    price: productPrice?.toString() ?? '0',
                    salePrice: (product_a?.price ?? '').toString(),
                    soldOut: product_a?.isSoldOut ?? false,
                };
                allProducts.push(product_b);
            }

            logInfo(
                functionName,
                `Fetched page=${page}/${totalPages} for brandId="${brandId}" (items=${products_a.length})`
            );
            // 페이지 간 딜레이 (너무 빠른 요청 방지)
            await delay(200 + 100 * Math.random());
        } catch (error) {
            logError(functionName, `Error on page ${page}: ${error.message}`);
            // 에러 발생 시에도 다음 페이지 시도
        }
    }

    // 3) 최종적으로 brand 정보 형태로 묶어 반환
    return {
        brandRepName,
        brandId,
        products: allProducts,
    };
}

/**
 * 무작위 User-Agent 헤더를 생성해서 반환
 */
function getRandomUserAgentHeader() {
    const userAgents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
        // 필요 시 다른 UA 추가
    ];
    const userAgent = userAgents[Math.floor(Math.random() * userAgents.length)];
    return {
        accept: 'application/json',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'ko-KR,ko;q=0.5',
        'sec-ch-ua': '"Chromium";v="130", "Brave";v="130", "Not?A_Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'sec-gpc': '1',
        'user-agent': userAgent,
    };
}

/**
 * 이 모듈 자체를 직접 실행하면,
 * ./musinsa/brands.json 을 읽고,
 * 각 브랜드별로 상품정보 수집 후
 * ./output/products/{brandId}.json 에 저장
 */
(async () => {
    const functionName = 'collectProductInfo_main';
    try {
        const brandsRaw = fs.readFileSync('./musinsa/brands.json', 'utf-8');
        const brands = JSON.parse(brandsRaw);

        for (const brand of brands) {
            const { brandId, brandRepName } = brand;
            const brandData = await collectProductInfo(brandId, brandRepName);

            // 저장 경로는 원하는대로 조정
            const outputPath = `./output/products/${brandId}.json`;
            fs.mkdirSync(`./output/products`, { recursive: true });
            fs.writeFileSync(outputPath, JSON.stringify(brandData, null, 2), 'utf-8');
            logInfo(functionName, `Saved product info => ${outputPath}`);
        }
    } catch (error) {
        logError(functionName, error.message);
    }
})();
