import os
import sys
import json
import argparse
import shutil
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 최소한의 모델 로딩 횟수를 위해 전역에서 lazy-import 식으로 모델 참조 (multiprocessing 시 child process마다 로드)
# YOLO 모듈 (ultralytics)와 Segformer 모듈 (transformers) 로드
# ----------------------------------------------------------------------------
from ultralyticsplus import YOLO, render_result
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from sklearn.cluster import KMeans
import colorsys
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)

# ----------------------------------------------------------------------------
# 전역 상수 & 라벨 정의
# ----------------------------------------------------------------------------
MIN_THRESHOLD = 0.3      # YOLO bbox 비율: 최소
MAX_THRESHOLD = 1.0      # YOLO bbox 비율: 최대
CONF_THRESHOLD = 0.7     # YOLO confidence 스코어 임계치
MIN_IMAGE_SIZE = 320     # YOLO 검증할 최소 이미지 크기
COLOR_SIMILARITY_THRESHOLD = 25.0  # CIEDE2000 유사도 임계값

id2label = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses",
    4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress",
    8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face",
    12: "Left-leg", 13: "Right-leg", 14: "Left-arm",
    15: "Right-arm", 16: "Bag", 17: "Scarf"
}

# ----------------------------------------------------------------------------
# 공통 유틸
# ----------------------------------------------------------------------------
def hex_to_lab(hex_color: str):
    """HEX -> CIE Lab 변환."""
    rgb = tuple(int(hex_color.lstrip('#')[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    rgb_color = sRGBColor(*rgb)
    return convert_color(rgb_color, LabColor)

def get_color_difference(hex1: str, hex2: str):
    """두 HEX 색상 사이의 CIEDE2000 차이값 계산."""
    color1_lab = hex_to_lab(hex1)
    color2_lab = hex_to_lab(hex2)
    return delta_e_cie2000(color1_lab, color2_lab)

def get_dominant_color(image_bgr: np.ndarray, mask: np.ndarray):
    """마스크 영역의 주요 색상을 k-means로 추출하여 (RGB, HEX) 형식으로 반환."""
    masked_img = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    pixels = masked_img[mask > 0].reshape(-1, 3)
    if len(pixels) == 0:
        return None, None

    kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]

    # BGR -> RGB -> HEX
    rgb_color = tuple(map(int, dominant_color[::-1]))
    hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
    return rgb_color, hex_color

def is_valid_detection(boxes, image_size):
    """
    YOLO detection 결과 중 (class=clothing=2) 박스가 있고,
    박스가 적절한 크기(비율)와 높은 confidence를 갖는 경우 valid 로 판단.
    """
    if len(boxes) == 0:
        return False

    img_w, img_h = image_size
    for box in boxes:
        # clothing 클래스(2) + confidence 확인
        if int(box.cls[0]) != 2:
            continue
        if float(box.conf[0]) < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bw = x2 - x1
        bh = y2 - y1
        w_ratio = bw / img_w
        h_ratio = bh / img_h

        # 하나라도 min~max 범위이면 유효
        if ((MIN_THRESHOLD <= w_ratio <= MAX_THRESHOLD) or 
            (MIN_THRESHOLD <= h_ratio <= MAX_THRESHOLD)):
            return True
    return False

# ----------------------------------------------------------------------------
# Step 1: YOLO 검출
# ----------------------------------------------------------------------------
def yolo_step(product_id: str, dataset_dir: str, output_dir: str):
    """
    해당 product_id에 대해 dataset_dir/wearing 하위의 'product_id_*.jpg' 파일을
    YOLO 모델로 검출 수행 -> valid / invalid 분류하고,
    결과를 output_dir/results/product_id 하위에 저장
    """
    model = YOLO('kesimeg/yolov8n-clothing-detection')
    model.overrides['verbose'] = False

    product_outdir = Path(output_dir) / 'results' / product_id
    product_outdir.mkdir(parents=True, exist_ok=True)

    results_data = {}
    valid_list = []
    invalid_list = []

    image_dir = Path(dataset_dir) / 'wearing'
    pattern = f"{product_id}_*.jpg"
    image_paths = list(image_dir.glob(pattern))

    for image_path in tqdm(image_paths, desc=f"[YOLO] {product_id}", leave=False):
        img = Image.open(image_path)
        img_size = img.size  # (width, height)

        # 너무 작은 이미지 제외
        if img_size[0] < MIN_IMAGE_SIZE and img_size[1] < MIN_IMAGE_SIZE:
            results_data[image_path.name] = {
                'is_valid': False,
                'boxes': [],
                'ratios': [],
                'validation_criteria': {
                    'min_threshold': MIN_THRESHOLD,
                    'max_threshold': MAX_THRESHOLD,
                    'conf_threshold': CONF_THRESHOLD,
                    'min_image_size': MIN_IMAGE_SIZE
                },
                'reason': 'Image size too small'
            }
            invalid_list.append(image_path.name)
            continue

        # 예측
        results = model.predict(str(image_path))
        boxes = results[0].boxes
        is_valid = is_valid_detection(boxes, img_size)

        # (clothing 클래스만) 비율 계산
        ratio_list = []
        for b in boxes:
            if int(b.cls[0]) == 2 and float(b.conf[0]) >= CONF_THRESHOLD:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                bw = x2 - x1
                bh = y2 - y1
                ratio_list.append({
                    'width_ratio': float(bw / img_size[0]),
                    'height_ratio': float(bh / img_size[1])
                })

        # 결과 저장
        results_data[image_path.name] = {
            'is_valid': is_valid,
            'boxes': [
                {
                    'coordinates': b.xyxy[0].tolist(),
                    'confidence': float(b.conf[0]),
                    'class': int(b.cls[0])
                }
                for b in boxes
                if int(b.cls[0]) == 2 and float(b.conf[0]) >= CONF_THRESHOLD
            ],
            'ratios': ratio_list,
            'validation_criteria': {
                'min_threshold': MIN_THRESHOLD,
                'max_threshold': MAX_THRESHOLD,
                'conf_threshold': CONF_THRESHOLD
            }
        }

        # 시각화 결과 저장
        render = render_result(model=model, image=str(image_path), result=results[0])
        render.save(str(product_outdir / f"visualized_{image_path.name}"))

        # valid/invalid 분류
        if is_valid:
            valid_list.append(image_path.name)
        else:
            invalid_list.append(image_path.name)

    # 상세 결과
    with open(product_outdir / "results_detail.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    # valid / invalid
    final_validation = {
        "valid": valid_list,
        "invalid": invalid_list
    }
    with open(product_outdir / "results.json", "w", encoding="utf-8") as f:
        json.dump(final_validation, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------------
# Step 2 & 3: 세그멘테이션 기반 검증(색상 비교 등) + 크롭
#    --> wearing 이미지 개수가 많다면 병렬처리로 가속
# ----------------------------------------------------------------------------
def validate_and_crop_step(product_id: str, dataset_dir: str, output_dir: str, worker: int = 1):
    """
    YOLO 결과(results.json)에서 valid로 분류된 wearing 이미지를 대상으로
    Segformer 세그멘테이션을 수행하여 (색상 유사도, dominant 카테고리 등) 검증하고,
    필요시 크롭 이미지까지 생성한다.
    
    최종적으로 "segmentation_results/product_id" 와 "cropped/product_id" 등 폴더에
    분석, 크롭 결과를 저장하고, 별도의 validation_results.json를 만들어
    추가적인 'valid/invalid' 분류도 수행한다.

    병렬처리(worker>1) 시, wearing 이미지 단위로 분할하여 병렬로 세그멘테이션/분석을 수행합니다.
    """
    # 먼저 Segformer 모델은 제품 이미지 분석을 위해 메인 프로세스에서만 1회 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
    model_seg = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes").to(device)

    # YOLO 결과(1차 valid/invalid) 불러오기
    results_json_path = Path(output_dir) / "results" / product_id / "results.json"
    if not results_json_path.exists():
        print(f"[validate_and_crop_step] {results_json_path} not found - skipping.")
        return
    with open(results_json_path, "r") as f:
        yolo_validation = json.load(f)
    valid_images = yolo_validation.get("valid", [])

    product_path = Path(dataset_dir) / "product" / f"{product_id}.jpg"
    if not product_path.exists():
        print(f"[validate_and_crop_step] product image {product_path} not found.")
        return

    # 결과 디렉토리 준비
    seg_outdir = Path(output_dir) / "segmentation_results" / product_id
    seg_outdir.mkdir(parents=True, exist_ok=True)
    crop_outdir = Path(output_dir) / "cropped" / product_id
    crop_outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # 보조 함수: 이미지 세그먼트 & confidence
    # ------------------------------------------------------------------------
    def process_single_image_local(image_path: Path):
        pil_img = Image.open(image_path).convert("RGB")
        inputs = processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model_seg(**inputs)
            logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=pil_img.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        probabilities = torch.nn.functional.softmax(upsampled_logits, dim=1)
        confidence, pred_seg = torch.max(probabilities, dim=1)
        return pil_img, pred_seg[0], confidence[0]

    def get_bbox_with_padding(mask, padding=30):
        y_inds, x_inds = np.where(mask > 0)
        if len(y_inds) == 0 or len(x_inds) == 0:
            return None
        x_min = max(0, np.min(x_inds) - padding)
        y_min = max(0, np.min(y_inds) - padding)
        x_max = min(mask.shape[1], np.max(x_inds) + padding)
        y_max = min(mask.shape[0], np.max(y_inds) + padding)
        return (x_min, y_min, x_max, y_max)

    # ------------------------------------------------------------------------
    # 1) product 이미지 세그멘테이션 (메인 프로세스에서 1회)
    # ------------------------------------------------------------------------
    product_img, product_seg, product_conf = process_single_image_local(product_path)
    product_array_bgr = cv2.cvtColor(np.array(product_img), cv2.COLOR_RGB2BGR)

    # dominant category (제품 이미지 상에서 가장 큰 파트를 차지하는)
    unique_ids, counts = np.unique(product_seg.cpu().numpy(), return_counts=True)
    category_counts = {id2label[i]: c for i, c in zip(unique_ids, counts) if i != 0}
    if not category_counts:
        dom_cat = None
    else:
        dom_cat = max(category_counts.items(), key=lambda x: x[1])[0]  # 예: 'Upper-clothes' 등
    label_id = None
    if dom_cat:
        label_id = [k for k, v in id2label.items() if v == dom_cat][0]

    # ------------------------------------------------------------------------
    # 2) valid 착용 이미지에 대해서 병렬 세그멘테이션 & validate + crop
    # ------------------------------------------------------------------------
    def analyze_wearing_image(wearing_image_name: str):
        """
        병렬 작업에서 실제로 호출될 함수.
        - 이 함수 안에서 다시 모델 로드(=비효율)할 수도 있으나,
          worker가 많지 않다면 허용 가능.
        - 더 효율적으로 하려면 'spawn' 모드에서 전역 모델 로딩 후 child프로세스 공유 기법도 가능.
        """
        # (각 프로세스에서) 모델 로드
        local_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
        _model_seg = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes").to(local_device)

        # wearing 이미지 열기
        wearing_path = Path(dataset_dir) / "wearing" / wearing_image_name
        if not wearing_path.exists():
            return None

        # 세그먼트
        pil_img = Image.open(wearing_path).convert("RGB")
        inputs = _processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(local_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = _model_seg(**inputs)
            logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=pil_img.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        probabilities = torch.nn.functional.softmax(upsampled_logits, dim=1)
        w_confidence, w_pred_seg = torch.max(probabilities, dim=1)
        wearing_seg = w_pred_seg[0]
        wearing_conf = w_confidence[0]
        wearing_array_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 색상 비교
        results_list = []
        for class_idx in range(1, 18):
            pmask = (product_seg == class_idx).cpu().numpy().astype(np.uint8)
            wmask = (wearing_seg == class_idx).cpu().numpy().astype(np.uint8)

            # 최소 픽셀수(1% 이상)일 때만
            pixel_thresh = pmask.size * 0.01
            if pmask.sum() <= pixel_thresh:
                continue

            pcolor_rgb, pcolor_hex = get_dominant_color(product_array_bgr, pmask)
            wcolor_rgb, wcolor_hex = get_dominant_color(wearing_array_bgr, wmask)
            if pcolor_hex and wcolor_hex:
                diff = get_color_difference(pcolor_hex, wcolor_hex)
                is_similar = diff < COLOR_SIMILARITY_THRESHOLD
                confidence_score = float(product_conf[pmask > 0].mean().item())
                pixel_ratio = round(pmask.sum() / pmask.size * 100, 2)

                results_list.append({
                    'category': id2label[class_idx],
                    'product_color': pcolor_hex,
                    'wearing_color': wcolor_hex,
                    'color_difference': round(diff, 2),
                    'is_similar': is_similar,
                    'confidence': round(confidence_score, 3),
                    'pixel_ratio': pixel_ratio,
                })

        results_list.sort(key=lambda x: x['pixel_ratio'], reverse=True)

        # Crop (dominant_category) 부분
        crop_result = None
        if label_id is not None:
            wmask_dom = (wearing_seg == label_id).cpu().numpy().astype(np.uint8)
            bbox = get_bbox_with_padding(wmask_dom)
            if bbox is not None:
                w_arr = np.array(pil_img)
                x_min, y_min, x_max, y_max = bbox
                cropped_arr = w_arr[y_min:y_max, x_min:x_max]
                crop_filename = wearing_image_name.replace(".jpg", "_crop.jpg")
                crop_path = crop_outdir / crop_filename
                Image.fromarray(cropped_arr).save(str(crop_path))

                crop_result = {
                    "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                    "cropped_image_path": str(crop_path),
                    "dominant_category_from_product": dom_cat
                }

        uid_part = wearing_image_name.split('_', 1)[-1]  # e.g. 3524616_123.jpg -> 123.jpg
        uid = uid_part.replace(".jpg", "")

        analysis_data = {
            "uid": uid,
            "wearing_image": str(wearing_path),
            "analysis_results": results_list,
            "dominant_category_from_product": dom_cat,
            "crop_result": crop_result
        }
        return analysis_data

    wearing_analyses = []
    if worker > 1 and len(valid_images) > 1:
        # 멀티프로세스
        with ProcessPoolExecutor(max_workers=worker, mp_context=multiprocessing.get_context("spawn")) as executor:
            tasks = {executor.submit(analyze_wearing_image, wi): wi for wi in valid_images}
            for future in tqdm(tasks, desc=f"[Validate+Crop] {product_id}", leave=True):
                result = future.result()
                if result is not None:
                    wearing_analyses.append(result)
    else:
        # 단일 프로세스 방식
        for wi in tqdm(valid_images, desc=f"[Validate+Crop] {product_id}", leave=True):
            res = analyze_wearing_image(wi)
            if res is not None:
                wearing_analyses.append(res)

    # 2차 valid/invalid
    validation_summary = {"valid": [], "invalid": []}
    for wa in wearing_analyses:
        wearing_image_bn = os.path.basename(wa["wearing_image"])
        # product의 dominant_category를 찾아서, is_similar=True 이면 valid
        dom_cat_result = None
        for r in wa["analysis_results"]:
            if r["category"] == wa["dominant_category_from_product"]:
                dom_cat_result = r
                break

        if dom_cat_result and dom_cat_result["is_similar"]:
            validation_summary["valid"].append(wearing_image_bn)
        else:
            validation_summary["invalid"].append(wearing_image_bn)

    # 결과 JSON 저장
    seg_outdir.mkdir(parents=True, exist_ok=True)
    valres_path = seg_outdir / "validation_results.json"
    with open(valres_path, "w", encoding="utf-8") as f:
        json.dump(validation_summary, f, indent=4)

    summary_data = {
        "product_id": product_id,
        "product_image": str(product_path),
        "dominant_category": dom_cat,
        "wearing_analyses": wearing_analyses
    }
    summary_json_path = seg_outdir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

# ----------------------------------------------------------------------------
# Step 4: arrange 정리
# ----------------------------------------------------------------------------
def arrange_step(product_id: str, dataset_dir: str, output_dir: str):
    """
    마지막으로 'cropped' / 'segmentation_results' / 'results' / 'output' 등에서
    필요한 파일을 모아 보고서를 생성하거나, 특정 디렉토리에 정렬할 수 있음.
    여기에서는 예시로:
    - 상위 단계에서 생성된 크롭 이미지를 output/wearing 으로 복사
    - product 이미지를 output/product 로 복사
    - 상위 6개 시각화 -> output/summary_{product_id}.png (arrange.py 참고)
    """
    from PIL import Image
    import matplotlib.gridspec as gridspec

    out_wearing = Path(output_dir) / "wearing"
    out_wearing.mkdir(parents=True, exist_ok=True)
    out_product = Path(output_dir) / "product"
    out_product.mkdir(parents=True, exist_ok=True)

    crop_dir = Path(output_dir) / "cropped" / product_id
    seg_dir = Path(output_dir) / "segmentation_results" / product_id
    product_img_path = Path(dataset_dir) / "product" / f"{product_id}.jpg"

    # 1) 크롭된 wearing 복사
    if crop_dir.exists():
        for img_name in os.listdir(crop_dir):
            if not img_name.endswith("_crop.jpg"):
                continue
            src = crop_dir / img_name
            dst = out_wearing / img_name
            shutil.copy2(src, dst)

    # 2) product 복사
    if product_img_path.exists():
        shutil.copy2(product_img_path, out_product / f"{product_id}.jpg")

    # 3) 요약(상위 6개) 시각화
    summary_json_path = seg_dir / "summary.json"
    if not summary_json_path.exists():
        return

    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    wearing_analyses = summary_data.get("wearing_analyses", [])

    # color_difference가 낮은 것이 상위
    scored_list = []
    for wa in wearing_analyses:
        min_diff = 999999
        for r in wa["analysis_results"]:
            if r["color_difference"] < min_diff:
                min_diff = r["color_difference"]
        scored_list.append((wa, min_diff))
    scored_list.sort(key=lambda x: x[1])
    selected = scored_list[:6]

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 6, height_ratios=[1, 1])

    # 첫 행: product
    if product_img_path.exists():
        ax_prod = plt.subplot(gs[0, :])
        pimg = Image.open(product_img_path)
        ax_prod.imshow(pimg)
        ax_prod.set_title(f"Product {product_id}\nDominant Category: {summary_data.get('dominant_category', 'N/A')}")
        ax_prod.axis("off")

    # 두 번째 행: top 6
    for idx, (wa, score) in enumerate(selected):
        if idx >= 6:
            break
        ax = plt.subplot(gs[1, idx])
        wear_bn = os.path.basename(wa["wearing_image"])
        cfn = wear_bn.replace(".jpg", "_crop.jpg")
        cfp = crop_dir / cfn
        if cfp.exists():
            cimg = Image.open(cfp)
            ax.imshow(cimg)
        ax.set_title(f"{wear_bn}\ncolor_diff={score:.1f}")
        ax.axis("off")

    plt.tight_layout()
    summary_fig_path = Path(output_dir) / f"summary_{product_id}.png"
    plt.savefig(str(summary_fig_path), bbox_inches='tight', dpi=200)
    plt.close()

# ----------------------------------------------------------------------------
# 최종 파이프라인 함수: product_id 단위 처리
# ----------------------------------------------------------------------------
def process_single_product(product_id: str, dataset_dir: str, output_dir: str, worker: int = 1):
    """
    1) YOLO - results.json
    2) validate & crop - validation_results.json, cropped/*, segmentation_results/*
       -> 멀티프로세스(seg) 분석 (worker>1)
    3) arrange (결과물 정리, summary 시각화)
    """
    yolo_step(product_id, dataset_dir, output_dir)
    validate_and_crop_step(product_id, dataset_dir, output_dir, worker)
    arrange_step(product_id, dataset_dir, output_dir)

# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="refine pipeline for clothing images.")
    parser.add_argument("--product_id", type=str, default=None, help="특정 product_id만 처리.")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="데이터셋 디렉토리 (product, wearing 폴더 포함).")
    parser.add_argument("--output_dir", type=str, default="output", help="결과물 저장할 루트 디렉토리.")
    parser.add_argument("--worker", type=int, default=1, help="병렬 (Seg) 처리 시 사용할 프로세스 수.")
    return parser.parse_args()

def main():
    args = parse_args()

    # product_id가 있으면 해당 id만 처리
    if args.product_id:
        product_ids = [args.product_id]
    else:
        # dataset_dir/product/*.jpg 중 파일명에서 product_id 추출
        prod_folder = Path(args.dataset_dir) / "product"
        product_ids = []
        for f in prod_folder.glob("*.jpg"):
            pid = f.stem  # 예: 2792167
            product_ids.append(pid)

    product_ids = sorted(list(set(product_ids)))

    # product_id 단위 멀티프로세스
    if len(product_ids) > 1 and args.worker > 1:
        with ProcessPoolExecutor(max_workers=args.worker, mp_context=multiprocessing.get_context("spawn")) as executor:
            list(tqdm(
                executor.map(lambda pid: process_single_product(pid, args.dataset_dir, args.output_dir, 1),
                             product_ids),
                total=len(product_ids),
                desc="[Refine Pipeline]"
            ))
    else:
        for pid in tqdm(product_ids, desc="[Refine Pipeline]"):
            process_single_product(pid, args.dataset_dir, args.output_dir, args.worker)

if __name__ == "__main__":
    # 다중프로세스 안정화를 위해
    multiprocessing.set_start_method("spawn", force=True)
    main()