import cv2
import numpy as np
import uuid
import requests
from datetime import datetime, timedelta
from google.cloud import storage
from typing import Optional, Tuple

from config import settings
from models import ai_manager
from schemas import (
    DiagnosticReport, YoloResult, CnnResult, 
    ProcessStatus, DiagnosisStatus
)

class ImageService:
    def __init__(self):
        # 初始化 GCS Client
        # Cloud Run 會自動抓取 Service Account，無需手動給 key
        self.storage_client = storage.Client(project=settings.GCP_PROJECT_ID)
        self.bucket = self.storage_client.bucket(settings.GCS_BUCKET_NAME)

    def _bytes_to_cv2(self, data: bytes) -> np.ndarray:
        """將 bytes 轉為 OpenCV 格式"""
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("無法解碼圖片，格式可能錯誤")
        return img

    def _cv2_to_bytes(self, image: np.ndarray) -> bytes:
        """將 OpenCV 圖片轉為 bytes (JPEG)"""
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("圖片編碼失敗")
        return buffer.tobytes()

    def _upload_to_gcs(self, image_data: bytes, folder: str, user_id: str) -> str:
        """
        上傳圖片至 GCS 並回傳 Public URL
        路徑格式: images/{folder}/{user_id}/{uuid}.jpg
        """
        filename = f"{uuid.uuid4()}.jpg"
        blob_path = f"images/{folder}/{user_id}/{filename}"
        blob = self.bucket.blob(blob_path)

        # 上傳
        blob.upload_from_string(image_data, content_type='image/jpeg')
        
        # 2. 產生 Signed URL
        try:
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(hours=24), # 設定連結有效時間(V4 簽章上限 7 天)
                method="GET"
            )
            return url
            
        except Exception as e:
            print(f"❌ Generate Signed URL failed: {e}")
            # 若簽章失敗 (例如本地缺乏 Service Account 金鑰)，回傳一個錯誤標示或空字串
            # 注意：若是在 Cloud Run，Service Account 需要有 'Service Account Token Creator' 角色
            return ""

    def _download_image_from_url(self, url: str) -> np.ndarray:
        """從 URL 下載圖片並轉為 OpenCV 格式 (用於 Phase 2)"""
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        nparr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    # ==========================================
    # 🚀 Phase 1: YOLO 偵測階段
    # ==========================================
    def run_yolo_phase(self, user_id: str, image_bytes: bytes) -> DiagnosticReport:
        # 1. 準備基礎資料
        report_id = str(uuid.uuid4())
        cv_image = self._bytes_to_cv2(image_bytes)

        # 2. 上傳原始圖片 (Original)
        original_url = self._upload_to_gcs(image_bytes, folder="original", user_id=user_id)

        # 3. 執行 YOLO 模型
        raw_yolo_output = ai_manager.yolo.predict(cv_image)

        # 強制轉換為 Schema 物件，確保資料結構正確
        if isinstance(raw_yolo_output, dict):
            yolo_result_obj = YoloResult(**raw_yolo_output)
        else:
            # 假設它已經是物件，直接使用 (或根據您的 ai_manager 實作調整)
            yolo_result_obj = raw_yolo_output

        # 4. 根據結果處理
        final_status = ProcessStatus.FAILED
        crop_url = None

        if yolo_result_obj.is_detected and yolo_result_obj.bbox:
            # --- 偵測到眼睛 ---
            x1, y1, x2, y2 = yolo_result_obj.bbox
            
            # 安全裁切 (避免超出邊界)
            h, w = cv_image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop_img = cv_image[y1:y2, x1:x2]
            
            # 上傳裁切圖 (Crop) -> 給使用者確認用
            if crop_img.size > 0:
                crop_bytes = self._cv2_to_bytes(crop_img)
                crop_url = self._upload_to_gcs(crop_bytes, folder="crops", user_id=user_id)
                
                # 更新 Result 物件內的 URL
                yolo_result_obj.crop_image_url = crop_url
                final_status = ProcessStatus.WAITING_USER
            else:
                # 裁切失敗 (極端情況)
                final_status = ProcessStatus.FAILED
        else:
            # --- 未偵測到眼睛 ---
            final_status = ProcessStatus.FAILED  # 或視為 Unknown 結束流程
            # 這裡我們保持 YOLO 結果為 False，讓前端顯示「未偵測到」

        # 5. 建立報告物件
        report = DiagnosticReport(
            report_id=report_id,
            user_id=user_id,
            current_status=final_status,
            original_image_url=original_url,
            yolo_result=yolo_result_obj,
            cnn_result=None, # Phase 1 還沒有 CNN 結果
            suggestion=None
        )

        return report

    # ==========================================
    # 🚀 Phase 2: CNN 診斷階段
    # ==========================================
    def run_cnn_phase(self, report: DiagnosticReport) -> DiagnosticReport:
        """
        接收使用者確認後的報告，下載裁切圖進行 CNN 分析
        """
        if not report.yolo_result or not report.yolo_result.crop_image_url:
            raise ValueError("報告缺少 YOLO 裁切圖，無法執行 CNN")

        # 1. 下載裁切圖片
        crop_img = self._download_image_from_url(report.yolo_result.crop_image_url)

        # 2. 執行 CNN 模型 (回傳 Result 物件 + 熱力圖圖片數據)
        cnn_result_obj, heatmap_img = ai_manager.cnn.predict(crop_img)

        # 3. 如果有熱力圖，上傳之
        heatmap_url = None
        if heatmap_img is not None:
            heatmap_bytes = self._cv2_to_bytes(heatmap_img)
            heatmap_url = self._upload_to_gcs(heatmap_bytes, folder="heatmaps", user_id=report.user_id)
            cnn_result_obj.heatmap_image_url = heatmap_url

        # 4. 更新報告
        report.cnn_result = cnn_result_obj
        report.current_status = ProcessStatus.COMPLETED
        
        # (可選) 在這裡簡單根據狀態給一些預設建議，或留給 LLM 層處理
        if cnn_result_obj.status == DiagnosisStatus.DETECTED:
            report.suggestion = "檢測到潛在高風險特徵，建議儘速就醫檢查。"
        
        return report

# 實例化 Service
image_service = ImageService()