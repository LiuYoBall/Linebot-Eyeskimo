from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # ==========================================
    # A. LINE Bot 設定 (必填，無預設值)
    # 程式啟動時若讀不到這兩項，會直接報錯停止 (Fail Fast)
    # ==========================================
    LINE_CHANNEL_SECRET: str
    LINE_CHANNEL_ACCESS_TOKEN: str

    # ==========================================
    # B. Google Cloud 設定 (必填)
    # ==========================================
    GCP_PROJECT_ID: str
    GCS_BUCKET_NAME: str  # 用來存使用者上傳圖與 AI 結果圖的 Bucket

    # ==========================================
    # C. AI 模型參數 (可透過環境變數微調)
    # ==========================================
    # 模型檔案路徑 (Container 內部的絕對路徑)
    MODEL_YOLO_PATH: str = "/app/weights/yolo.pt"
    MODEL_CNN_PATH: str = "/app/weights/CNN.pth"

    # 判定邏輯 (Thresholds)
    AI_CONF_THRESHOLD: float = 0.25  # YOLO 信心門檻
    
    # 雙重閥值邏輯:
    # Prob < LOW        -> Not-Detected
    # LOW <= Prob < HIGH -> Risk
    # Prob >= HIGH      -> Detected
    AI_THRESH_LOW: float = 0.4       
    AI_THRESH_HIGH: float = 0.75

    # ==========================================
    # D. 其他設定
    # ==========================================
    # 是否開啟除錯模式 (本地開發設 True, 雲端設 False)
    DEBUG_MODE: bool = False
    
    # 允許上傳的圖片格式
    ALLOWED_EXTENSIONS: set = {'.jpg', '.jpeg', '.png', '.bmp'}

    # Pydantic 設定：指定讀取 .env 檔案 (僅本地開發有效)
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore" # 忽略多餘的變數
    )

    # ==========================================
    # E. LLM 設定 (Groq / OpenAI)
    # ==========================================
    OPENAI_API_KEY: Optional[str] = None
    
    # 若使用 Groq，填入: https://api.groq.com/openai/v1
    # 若使用 OpenAI，填入: https://api.openai.com/v1 (或留空)
    OPENAI_BASE_URL: str = "https://api.groq.com/openai/v1"
    
    # 指定模型名稱 (Groq 推薦: llama-3.3-70b-versatile)
    LLM_MODEL: str = "llama-3.3-70b-versatile"

# 實例化設定物件，其他檔案只要 import 這個 settings 即可
settings = Settings()