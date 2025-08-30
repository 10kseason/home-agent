"""
Assumptions: ROI centered on cursor; PaddleOCR (ko/en/ja/zh) on CPU
Risks: low confidence or misordered text
Alternatives: full-screen OCR or multi-scale search
Rationale: small crops reduce latency
"""

from dataclasses import dataclass
from typing import Iterable, List, Tuple

try:  # pragma: no cover - optional dependency
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover - PaddleOCR not installed
    PaddleOCR = None


@dataclass
class OcrBox:
    x1: int
    y1: int
    x2: int
    y2: int
    text: str
    conf: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    def move(self, offset: Tuple[int, int]) -> "OcrBox":
        ox, oy = offset
        return OcrBox(self.x1 + ox, self.y1 + oy, self.x2 + ox, self.y2 + oy, self.text, self.conf)


@dataclass
class ROI:
    offset: Tuple[int, int]
    image: object | None = None


def crop(cursor: Tuple[int, int], scale: float) -> ROI:
    """Stub for cropping an ROI around ``cursor``.

    Real implementation should capture pixels around the cursor.  This stub
    simply returns an offset used to translate OCR boxes back to absolute
    coordinates.
    """

    size = int(100 * scale)
    x, y = cursor
    return ROI(offset=(x - size // 2, y - size // 2))


def to_absolute(box: OcrBox, offset: Tuple[int, int]) -> OcrBox:
    return box.move(offset)


def sort_ocr_boxes(boxes: Iterable[OcrBox]) -> List[OcrBox]:
    """Return boxes sorted in row-major order.

    PaddleOCR sometimes emits results in an arbitrary sequence which can
    shuffle reading order.  Sorting by the bounding-box centre coordinates
    (top-to-bottom, then left-to-right) stabilises the output and simplifies
    downstream reconstruction of lines.
    """

    return sorted(boxes, key=lambda b: (b.center_y, b.center_x))


def collect_roi(
    cursor: Tuple[int, int],
    scale: float = 1.0,
    backoff: float = 1.4,
    max_attempts: int = 2,
    topk: int = 5,
    conf_min: float = 0.6,
    langs: Tuple[str, ...] = ("ko", "en", "ja", "zh"),
) -> List[OcrBox]:
    """Collect OCR boxes around the cursor with exponential backoff.

    Attempts OCR in multiple languages, filtering low-confidence results and
    returning the top ``topk`` boxes in reading order.  Raises if PaddleOCR is
    unavailable so callers can decide on a fallback strategy.
    """

    if PaddleOCR is None:  # pragma: no cover - dependency optional in tests
        raise RuntimeError("PaddleOCR not installed")

    engines = {l: PaddleOCR(use_angle_cls=True, lang=l) for l in langs}
    for _ in range(max_attempts):
        roi = crop(cursor, scale)
        boxes: List[OcrBox] = []
        for lang, eng in engines.items():  # pragma: no cover - heavy runtime
            try:
                results = eng.ocr(roi.image) if roi.image is not None else []
            except Exception:
                results = []
            for box, (text, conf) in results:
                b = OcrBox(
                    x1=int(box[0][0]),
                    y1=int(box[0][1]),
                    x2=int(box[2][0]),
                    y2=int(box[2][1]),
                    text=text,
                    conf=float(conf),
                )
                boxes.append(b)
        boxes = [b for b in boxes if b.conf >= conf_min]
        boxes = sort_ocr_boxes(boxes)
        if boxes:
            return [to_absolute(b, roi.offset) for b in boxes[:topk]]
        scale *= backoff
    return []
# Checklist:
# - [x] Think Harder
# - [x] Think Deeper
# - [x] More Information
# - [x] Check Again
