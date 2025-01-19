import fs from 'fs';

export function logInfo(functionName, message) {
    console.log(`[${new Date().toISOString()}] [${functionName}] [Info] ${message}`);
}

export function logError(functionName, message) {
    console.error(`[${new Date().toISOString()}] [${functionName}] [Error] ${message}`);
}

/**
 * 지정된 밀리초(ms)만큼 기다리는 함수
 * @param {number} ms
 * @returns {Promise<void>}
 */
export function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * 0.5초 ~ 1초 사이 랜덤 딜레이
 * @returns {Promise<void>}
 */
export async function randomShortDelay() {
    const randomMs = 500 + Math.random() * 500; // 0.5초 ~ 1초
    return delay(randomMs);
}

export function saveLog(data) {
    const logLine = `[${new Date().toISOString()}]\n${data}\n`;
    fs.appendFileSync('debug.log', logLine);
}
