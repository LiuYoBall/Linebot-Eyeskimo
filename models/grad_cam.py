import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCamGenerator:
    @staticmethod
    def generate(features, gradients, target_size):
        """
        features: CNN 最後一層卷積輸出 (Tensor)
        gradients: 反向傳播算出的梯度 (Tensor)
        target_size: (height, width) 原始裁切圖大小
        """
        if features is None or gradients is None:
            return None

        # 1.計算權重 (Global Average Pooling on Gradients)
        # pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # 2.加權特徵圖
        activation_map = features[0].clone()
        for i in range(activation_map.shape[0]):
            activation_map[i, :, :] *= pooled_gradients[i]
            
        # 3.產生熱力圖 (ReLU 濾除負值)
        heatmap = torch.mean(activation_map, dim=0)
        heatmap = F.relu(heatmap)
        
        # 4.正規化 (0~1)
        if torch.max(heatmap) != 0:
            heatmap /= torch.max(heatmap)
            
        # 5.轉為 Numpy 並調整大小
        heatmap_np = heatmap.cpu().detach().numpy()
        heatmap_resized = cv2.resize(heatmap_np, (target_size[1], target_size[0]))
        
        # 6.轉為彩色熱力圖 (JET Colormap)
        heatmap_color = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
        
        return heatmap_color