import json
from pathlib import Path
from typing import Dict, Any

from linebot import LineBotApi, WebhookHandler
from linebot.models import RichMenu, RichMenuSize, RichMenuArea, RichMenuBounds, MessageAction
from linebot.models import (
    TextSendMessage, FlexSendMessage, 
    QuickReply, QuickReplyButton, PostbackAction
)
from linebot.exceptions import LineBotApiError

from config import settings
from schemas import DiagnosticReport, DiagnosisStatus

class LineService:
    def __init__(self):
        # 初始化 LINE Bot API
        self.api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)
        self.handler = WebhookHandler(settings.LINE_CHANNEL_SECRET)
        
        # 定義多個資源路徑
        self.base_dir = Path("assets")
        self.template_dir = self.base_dir / "templates"
        self.knowledge_dir = self.base_dir / "knowledge" / "static_cards"

        # 載入主題設定 (若檔案不存在需有防呆)
        theme_path = self.base_dir / "styles" / "themes.json"
        self.themes = self._load_json(theme_path) if theme_path.exists() else {}

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """通用 JSON 讀取工具"""
        if not path.exists():
            print(f"⚠️ Warning: File not found: {path}")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Load JSON failed: {e}")
            return {}

    def _load_template(self, filename: str) -> Dict[str, Any]:
        """讀取 JSON 模板並回傳 Dict"""
        # 1. 嘗試從 UI 樣板目錄讀取
        path = self.template_dir / filename
        if path.exists():
            return self._load_json(path)
        
        # 2. 嘗試從 衛教知識卡片目錄 讀取
        path = self.knowledge_dir / filename
        if path.exists():
            return self._load_json(path)
        
        # 3. 都找不到，拋出錯誤
        raise FileNotFoundError(f"Template/Card not found: {filename}")

    def reply_text(self, reply_token: str, text: str):
        """回覆純文字"""
        # try:
        #     self.api.reply_message(reply_token, TextSendMessage(text=text))
        # except Exception as e:
        #     print(f"❌ Reply text failed: {e}")

        print(f"DEBUG: 準備回覆 Token: {reply_token}，內容: {text}")
        self.api.reply_message(reply_token, TextSendMessage(text=text))
        print("DEBUG: 回覆成功！")

    # ==========================================
    # 🚀 Phase 1: 發送 YOLO 確認卡片
    # ==========================================
    def send_crop_confirmation(self, reply_token: str, report: DiagnosticReport):
        """
        發送 YOLO 裁切結果，請求使用者確認
        """
        # 1. 基本防呆
        if not report.yolo_result or not report.yolo_result.crop_image_url:
            print("❌ No crop image to confirm.")
            self.reply_text(reply_token, "無法偵測到眼睛，請重新拍攝。")
            return

        try:
            # 2. 讀取 JSON 樣板
            template_name = "crop_confirmation.json"
            bubble = self._load_template(template_name)
            
            # 3. 轉換為字串以進行變數替換
            json_str = json.dumps(bubble)
            
            # (A) 替換圖片連結
            json_str = json_str.replace("PLACEHOLDER_CROP_IMG", report.yolo_result.crop_image_url)
            
            # (B) 替換 Report ID (讓 Postback 帶回正確的 ID)
            json_str = json_str.replace("PLACEHOLDER_REPORT_ID", report.report_id)
            
            # 4. 轉回 JSON 物件並發送
            final_bubble = json.loads(json_str)
            
            self.api.reply_message(
                reply_token,
                FlexSendMessage(alt_text="請確認眼睛偵測範圍", contents=final_bubble)
            )
            
        except Exception as e:
            print(f"❌ Push confirmation failed: {e}")
            # JSON 讀取失敗，回傳一個純文字 Fallback
            self.reply_text(reply_token, "眼睛位置偵測完成，請確認是否進行分析？")

    # ==========================================
    # 🚀 Phase 2: 發送最終診斷報告
    # ==========================================
    def send_analysis_result(self, reply_token: str, report: DiagnosticReport):
        if not report.cnn_result:
            self.reply_text(reply_token, "分析失敗，無結果。")
            return

        cnn = report.cnn_result
        
        # 1. 準備共用變數
        # 若是正常，可能沒有熱力圖，這時就用裁切圖當主圖
        img_main = cnn.heatmap_image_url if cnn.heatmap_image_url else report.yolo_result.crop_image_url
        img_sub1 = report.yolo_result.crop_image_url
        img_sub2 = report.original_image_url
        report_id_short = report.report_id[:8] # 取前8碼顯示即可

        try:
            # 2. 分流：決定使用哪個樣板
            if cnn.status == DiagnosisStatus.NOT_DETECTED:
                # === 正常流程 (Normal) ===
                template_name = "result_normal.json"
                
                # 讀取
                bubble = self._load_template(template_name)
                json_str = json.dumps(bubble)
                
                # 替換變數 (Normal 只需要換主圖和 ID)
                json_str = json_str.replace("PLACEHOLDER_IMG_MAIN", img_main)
                json_str = json_str.replace("PLACEHOLDER_REPORT_ID", report_id_short)
                
                # 預設主題
                theme = self.themes.get("default", {})

            else:
                # === 異常流程 (Warning) ===
                template_name = "result_warning.json"
                
                # 取得主題色設定 (從 themes.json)
                disease_key = cnn.disease if cnn.disease in self.themes else "default"
                theme = self.themes.get(disease_key, self.themes["default"])
                
                # 讀取
                bubble = self._load_template(template_name)
                json_str = json.dumps(bubble)

                # 替換圖片 (Warning 需要三張圖)
                json_str = json_str.replace("PLACEHOLDER_IMG_MAIN", img_main)
                json_str = json_str.replace("PLACEHOLDER_IMG_SUB1", img_sub1)
                json_str = json_str.replace("PLACEHOLDER_IMG_SUB2", img_sub2)
                # 替換 ID
                json_str = json_str.replace("PLACEHOLDER_REPORT_ID", report_id_short)

                # 替換主題顏色與文字
                for key, value in theme.items():
                    json_str = json_str.replace(f"PLACEHOLDER_{key}", value)
                
                # 4. 根據疾病名稱替換問卷觸發指令
                # 必須對應 main.py handle_text_message 邏輯
                survey_map = {
                    "Cataract": "白內障檢測",
                    "Conjunctivitis": "結膜炎檢測"
                }
                # 若找不到對應疾病，預設導向主選單的問診模式
                survey_cmd = survey_map.get(cnn.disease, "文字問診模式")
                
                json_str = json_str.replace("PLACEHOLDER_SURVEY_CMD", survey_cmd)

            # 3. 發送訊息
            # 為了讓標題好看，若有 disease_name 就顯示，沒有就顯示預設文字
            alt_text_title = theme.get('DISEASE_NAME', '檢測結果') if cnn.status != DiagnosisStatus.NOT_DETECTED else "檢測正常"
            
            self.api.reply_message(
                reply_token,
                FlexSendMessage(
                    alt_text=f"分析報告：{alt_text_title}", 
                    contents=json.loads(json_str)
                )
            )
            
        except Exception as e:
            print(f"❌ Send analysis result failed: {e}")
            self.reply_text(reply_token, "產生報告時發生錯誤。")

    def send_question(self, reply_token: str, question_data: dict):
        """
        發送原生 LINE JSON 格式的問卷題目
        """
        try:
            # 1. 取得題目文字
            text = question_data.get("text", "請回答以下問題")

            # 取得該題目的流程控制參數
            survey_id = question_data.get("survey")
            next_id = question_data.get("next")
            
            # 2. 處理 QuickReply
            qr_items_json = question_data.get("quickReply", {}).get("items", [])
            
            quick_reply_buttons = []
            for item in qr_items_json:
                action_data = item.get("action", {})
                
                # 取得原本的 data 
                original_data = action_data.get("data", "")
                
                # 將 survey 和 next 自動拼接到 data 後面
                new_data = f"{original_data}&survey={survey_id}&next={next_id}"

                # 建立 PostbackAction
                action = PostbackAction(
                    label=action_data.get("label"),
                    data=new_data,
                    display_text=action_data.get("displayText") # 讓使用者點擊後會說話
                )
                quick_reply_buttons.append(QuickReplyButton(action=action))

            # 3. 組合並發送
            if quick_reply_buttons:
                message = TextSendMessage(
                    text=text,
                    quick_reply=QuickReply(items=quick_reply_buttons)
                )
                self.api.reply_message(reply_token, message)
            else:
                # 萬一沒有按鈕，就只傳文字
                self.reply_text(reply_token, text)
                
        except Exception as e:
            print(f"❌ Send question failed: {e}")

    # Helper 函式來動態產生「清單內容」
    def send_history_list(self, reply_token: str, records: list):
        """
        發送歷史紀錄
        """
        try:
            # 1. 讀取主框架 (Container)
            bubble = self._load_template("history_list.json")
            
            if not records:
                # 若無紀錄，替換提示文字
                json_str = json.dumps(bubble).replace("PLACEHOLDER_EMPTY_MSG", "目前尚無檢查紀錄。")
                final_bubble = json.loads(json_str)
            else:
                # 2. 讀取單列樣板 (Item Template)
                row_template = self._load_template("history_row.json")
                row_template_str = json.dumps(row_template)
                
                content_box = []
                
                # 3. 動態生成 (Loop & Replace)
                for rec in records:
                    # 複製樣板字串並替換變數
                    current_row_str = row_template_str \
                        .replace("PLACEHOLDER_DATE", rec["date"]) \
                        .replace("PLACEHOLDER_STATUS", rec["status"]) \
                        .replace("PLACEHOLDER_COLOR", rec["color"]) \
                        .replace("PLACEHOLDER_REPORT_ID", rec["id"])
                    
                    # 轉回 Dict 並加入列表
                    content_box.append(json.loads(current_row_str))
                    
                    # 加入分隔線 (可選: 最後一筆不要分隔線)
                    content_box.append({"type": "separator", "margin": "sm"})
                
                # 移除最後多餘的分隔線 (Pythonic way)
                if content_box and content_box[-1]["type"] == "separator":
                    content_box.pop()

                # 4. 組合回主框架
                # 注意：history_list.json 的 body contents 預設可能有一個 placeholder 物件，直接覆蓋掉
                bubble["body"]["contents"] = content_box
                final_bubble = bubble

            # 5. 發送
            self.api.reply_message(
                reply_token,
                FlexSendMessage(alt_text="歷史檢查紀錄", contents=final_bubble)
            )

        except Exception as e:
            print(f"❌ Send history failed: {e}")
            self.reply_text(reply_token, "查詢紀錄時發生錯誤。")


# 實例化
line_service = LineService()