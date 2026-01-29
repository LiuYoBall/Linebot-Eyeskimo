import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

from config import settings
from schemas import CnnResult, DiagnosisStatus, DiseaseType
from models.grad_cam import GradCamGenerator

class ClassifyModel:
    def __init__(self):
        print(f"🔄 Loading DenseNet model from {settings.MODEL_CNN_PATH}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化 DenseNet121 並修改分類層
        try:
            self.model = models.densenet121(weights=None)
        except:
            self.model = models.densenet121(pretrained=False)
            
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 2)
        
        # 載入權重
        checkpoint = torch.load(settings.MODEL_CNN_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # 預處理流程
        self.preprocess_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _resize_with_gray_padding(self, image, target_size=(224, 224)):
        """ 保持比例縮放並補灰色邊"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized_image = cv2.resize(image, (new_w, new_h))
        canvas = np.full((target_h, target_w, 3), 127, dtype=np.uint8)
        
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
        return canvas

    def predict(self, crop_image: np.ndarray) -> CnnResult:
        # 1. 預處理
        padded_img = self._resize_with_gray_padding(crop_image)
        pil_img = Image.fromarray(cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB))
        input_tensor = self.preprocess_transform(pil_img).unsqueeze(0).to(self.device)
        
        # 2. 開啟梯度追蹤 (為了 Grad-CAM)
        input_tensor.requires_grad_()
        
        # 3. 前向傳播 (手動執行 features 層以掛載 hook)
        self.model.zero_grad()
        features = self.model.features(input_tensor)
        features.retain_grad() # 關鍵 hook
        
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        outputs = self.model.classifier(out)
        probs = torch.sigmoid(outputs)[0]
        
        # 4. 解析結果
        p_cat = probs[0].item()
        p_conj = probs[1].item()
        
        if p_cat > p_conj:
            dominant_prob = p_cat
            disease_enum = DiseaseType.CATARACT
            target_idx = 0
        else:
            dominant_prob = p_conj
            disease_enum = DiseaseType.CONJUNCTIVITIS
            target_idx = 1
            
        # 5. 判定狀態 (雙重閥值)
        # 使用設定檔中的閥值
        if dominant_prob >= settings.AI_THRESH_HIGH:
            status = DiagnosisStatus.DETECTED
        elif dominant_prob >= settings.AI_THRESH_LOW:
            status = DiagnosisStatus.RISK
        else:
            status = DiagnosisStatus.NOT_DETECTED
            
        # 6. 生成 Grad-CAM (僅 Risk/Detected 需要)
        heatmap_img = None
        if status in [DiagnosisStatus.RISK, DiagnosisStatus.DETECTED]:
            # 反向傳播計算梯度
            outputs[0, target_idx].backward()
            gradients = features.grad
            
            # 生成熱力圖 (BGR格式)
            raw_heatmap = GradCamGenerator.generate(features, gradients, crop_image.shape[:2])
            
            # 疊加圖片 (0.6 原圖 + 0.4 熱力圖)
            # 這裡我們回傳疊加好的圖，方便 Service 直接存
            if raw_heatmap is not None:
                heatmap_img = cv2.addWeighted(crop_image, 0.6, raw_heatmap, 0.4, 0)

        return CnnResult(
            status=status,
            disease=disease_enum,
            confidence=dominant_prob,
            prob_cataract=p_cat,
            prob_conjunctivitis=p_conj,
            heatmap_image_url=None, # 這裡先給 None，Service 層存圖後會填入
        ), heatmap_img # 多回傳一個 image data 給 Service 存