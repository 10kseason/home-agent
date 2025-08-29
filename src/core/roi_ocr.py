"""
Assumptions: ROI centered on cursor; PaddleOCR (ko/en/ja/zh) on CPU
Risks: low confidence or misordered text
Alternatives: full-screen OCR or multi-scale search
Rationale: small crops reduce latency
"""

def collect_roi(cursor, scale=1.0, backoff=1.4, max_attempts=2, topk=5, conf_min=0.6, langs=("ko", "en", "ja", "zh")):
    engines = {l: PaddleOCR(use_angle_cls=True, lang=l) for l in langs}
    for _ in range(max_attempts):
        roi = crop(cursor, scale)
        boxes = []
        for l in langs:
            boxes.extend(engines[l].ocr(roi))
        boxes = [b for b in boxes if b.conf >= conf_min]
        boxes.sort(key=lambda b: (-b.conf, b.center_y, b.center_x))
        if boxes:
            return [to_absolute(b, roi.offset) for b in boxes[:topk]]
        scale *= backoff
    return []
# Checklist:
# - [x] Think Harder
# - [x] Think Deeper
# - [x] More Information
# - [x] Check Again
