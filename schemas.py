from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
import time

# ==========================================
# 1. 列舉定義 (Enums)
# ==========================================
class DiagnosisStatus(str, Enum):
    NOT_DETECTED = "Not-Detected" # 未檢出 (低於 Risk 門檻)
    RISK = "Risk"                 # 風險 (介於兩門檻之間)
    DETECTED = "Detected"         # 確診/高風險 (高於 Detected 門檻)
    UNKNOWN = "Unknown"           # YOLO 未偵測到眼睛 (需重拍)

class ProcessStatus(str, Enum):
    PROCESSING_YOLO = "processing_yolo"   # 階段一：正在跑 YOLO
    WAITING_USER = "waiting_for_user"     # 階段一完成：等待使用者確認裁切圖
    PROCESSING_CNN = "processing_cnn"     # 階段二：使用者已確認，正在跑 CNN
    COMPLETED = "completed"               # 全部完成
    FAILED = "failed"                     # 發生錯誤

class DiseaseType(str, Enum):
    CATARACT = "Cataract"
    CONJUNCTIVITIS = "Conjunctivitis"
    NONE = "None"

# ==========================================
# 2. 階段一：YOLO 偵測結果
# ==========================================
class YoloResult(BaseModel):
    is_detected: bool = Field(..., description="是否偵測到眼睛物件")
    confidence: float = Field(..., description="YOLO 信心分數")
    
    # 座標 [x1, y1, x2, y2]，若沒抓到則為 None
    bbox: Optional[List[int]] = Field(None, description="眼睛框選座標")
    
    # 用於給使用者確認的圖片 (GCS URL)
    crop_image_url: Optional[str] = Field(None, description="裁切後的預覽圖連結")

# ==========================================
# 3. 階段二：CNN 診斷結果
# ==========================================
class CnnResult(BaseModel):
    status: DiagnosisStatus = Field(..., description="最終判讀狀態")
    disease: DiseaseType = Field(..., description="主要判定的病徵")
    confidence: float = Field(..., description="主要病徵的機率值")
    
    # 詳細機率 (用於繪製長條圖)
    prob_cataract: float
    prob_conjunctivitis: float
    
    # Grad-CAM 熱力圖連結 (僅 Risk/Detected 會有)
    heatmap_image_url: Optional[str] = None 

# ==========================================
# 4. 完整診斷報告 (Database Schema)
# ==========================================
class DiagnosticReport(BaseModel):
    report_id: str = Field(..., description="報告唯一識別碼 (UUID)")
    user_id: str = Field(..., description="LINE User ID")
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    
    # --- 流程控制 ---
    current_status: ProcessStatus = Field(..., description="目前流程進度")
    
    # --- 原始資源 ---
    original_image_url: str = Field(..., description="使用者上傳的原始圖")
    
    # --- 分段結果 (Optional) ---
    yolo_result: Optional[YoloResult] = None
    cnn_result: Optional[CnnResult] = None
    
    # --- 最終建議 (LLM / 規則產生) ---
    suggestion: Optional[str] = None