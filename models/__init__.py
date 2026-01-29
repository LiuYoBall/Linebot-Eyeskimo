from .segmentation import SegmentationModel
from .classify import ClassifyModel

# ==========================================
# AI Model Singleton Manager
# ==========================================
class AIModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            print("🚀 Initializing AI Models (Singleton)...")
            cls._instance = super(AIModelManager, cls).__new__(cls)
            
            # 在這裡初始化模型，保證全域只執行一次
            cls._instance.segmentation = SegmentationModel()
            cls._instance.classifier = ClassifyModel()
            
            print("✅ AI Models loaded ready.")
        return cls._instance

    @property
    def yolo(self) -> SegmentationModel:
        return self.segmentation

    @property
    def cnn(self) -> ClassifyModel:
        return self.classifier

# 全域變數：外部只要 import 這個變數即可使用
ai_manager = AIModelManager()