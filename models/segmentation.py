import cv2
import numpy as np
from ultralytics import YOLO
from config import settings
from schemas import YoloResult

class SegmentationModel:
    def __init__(self):
        print(f"🔄 Loading YOLO model from {settings.MODEL_YOLO_PATH}...")
        # 載入模型 (通常在 app 啟動時執行一次)
        self.model = YOLO(settings.MODEL_YOLO_PATH)

    def predict(self, image: np.ndarray) -> YoloResult:
        """
        輸入: OpenCV BGR 圖片 (numpy array)
        輸出: YoloResult Pydantic 物件
        """
        # 執行推論
        results = self.model.predict(
            image, 
            conf=settings.AI_CONF_THRESHOLD, 
            verbose=False
        )
        result = results[0]

        # 判斷是否偵測到物件
        if len(result.boxes) == 0:
            return YoloResult(
                is_detected=False,
                confidence=0.0,
                bbox=None,
                crop_image_url=None
            )

        # 取出信心分數最高的 Box
        best_conf_idx = int(result.boxes.conf.argmax())
        box = result.boxes[best_conf_idx]
        
        # 取得座標 [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        return YoloResult(
            is_detected=True,
            confidence=conf,
            bbox=[x1, y1, x2, y2],
            crop_image_url=None # URL 由 Service 層上傳 GCS 後填入
        )