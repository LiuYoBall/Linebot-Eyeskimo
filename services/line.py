import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    TextSendMessage, FlexSendMessage, 
    QuickReply, QuickReplyButton, PostbackAction,
    CameraAction, CameraRollAction
)

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
    
    def send_camera_request(self, reply_token: str):
        """
        發送 Quick Reply 引導使用者開啟相機或相簿
        """
        try:
            message = TextSendMessage(
                text="請選擇上傳方式，或直接傳送一張眼睛照片 📸",
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(action=CameraAction(label="開啟相機")),
                        QuickReplyButton(action=CameraRollAction(label="選擇相片"))
                    ]
                )
            )
            self.api.reply_message(reply_token, message)
            print(f"DEBUG: 已發送相機引導按鈕 Token: {reply_token}")
            
        except Exception as e:
            print(f"❌ Send camera request failed: {e}")
            # 失敗時的備案
            self.reply_text(reply_token, "請直接上傳一張眼睛照片。")

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
        report_id_short = report.report_id[:8] # 取前8碼

        try:
            # 取得狀態與疾病的鍵值
            status_str = cnn.status.name if hasattr(cnn.status, "name") else str(cnn.status)
            disease_val = cnn.disease.value if hasattr(cnn.disease, "value") else str(cnn.disease)

            # 分流：決定使用哪個樣板
            if status_str == "NOT_DETECTED":
                # === 正常流程 (Normal) ===
                template_name = "result_normal.json"
                bubble = self._load_template(template_name)
                json_str = json.dumps(bubble)
                
                # 準備圖片連結 
                boxed_url = report.original_boxed_url if report.original_boxed_url else report.original_image_url
                crop_url = report.yolo_result.crop_image_url
                
                # 執行替換
                json_str = json_str.replace("PLACEHOLDER_IMG_ORIGINAL_BOXED", boxed_url)
                json_str = json_str.replace("PLACEHOLDER_IMG_CROP", crop_url)
                json_str = json_str.replace("PLACEHOLDER_REPORT_ID", report_id_short)
                
                # 正常狀態的推播預覽文字
                alt_text_title = "檢測無明顯異常"

            else:
                # === 異常流程 (Warning) ===
                template_name = "result_warning.json"
            
                # 為了防止設定檔讀不到，給予預設值防呆
                theme_dict = self.themes.get("status_themes", {})
                disease_dict = self.themes.get("disease_info", {})

                theme_config = theme_dict.get(status_str, theme_dict.get("default", {}))
                disease_config = disease_dict.get(disease_val, disease_dict.get("default", {}))

                # 動態組合標題
                dynamic_title = theme_config.get("TITLE_TEMPLATE", "").format(disease=disease_config.get("name", ""))

                # 讀取並替換 JSON 樣板
                bubble = self._load_template(template_name)
                json_str = json.dumps(bubble)

                # --- 圖片替換區塊維持不變 ---
                # A. 框選原圖 (左上)
                boxed_url = report.original_boxed_url if report.original_boxed_url else report.original_image_url
                json_str = json_str.replace("PLACEHOLDER_IMG_ORIGINAL_BOXED", boxed_url)

                # B. 裁切圖 (右上)
                crop_url = report.yolo_result.crop_image_url
                json_str = json_str.replace("PLACEHOLDER_IMG_CROP", crop_url)

                # C. 熱力圖 CAM (左下)
                cam_url = cnn.heatmap_image_url if cnn.heatmap_image_url else crop_url
                json_str = json_str.replace("PLACEHOLDER_IMG_CAM", cam_url)

                # D. 直方圖 Chart (右下)
                chart_url = cnn.chart_image_url if cnn.chart_image_url else "https://via.placeholder.com/300?text=No+Chart"
                json_str = json_str.replace("PLACEHOLDER_IMG_CHART", chart_url)

                # E. ID替換
                json_str = json_str.replace("PLACEHOLDER_REPORT_ID", report_id_short)

                # -----------------------------
                # 替換視覺與文字設定
                json_str = json_str.replace("PLACEHOLDER_HEADER_BG", theme_config.get("HEADER_BG", "#F5F5F5"))
                json_str = json_str.replace("PLACEHOLDER_BADGE_BG", theme_config.get("BADGE_BG", "#EEEEEE"))
                json_str = json_str.replace("PLACEHOLDER_BADGE_TEXT", theme_config.get("BADGE_TEXT", "檢測結果"))
                json_str = json_str.replace("PLACEHOLDER_THEME_COLOR", theme_config.get("THEME_COLOR", "#666666"))

                json_str = json_str.replace("PLACEHOLDER_TITLE", dynamic_title)
                json_str = json_str.replace("PLACEHOLDER_DISEASE_NAME", disease_config.get("topic", "prevention"))
                json_str = json_str.replace("PLACEHOLDER_SURVEY_CMD", disease_config.get("survey_cmd", "文字問診模式"))

                # 推播預覽文字
                alt_text_title = dynamic_title

            # 發送訊息 (共用區塊)
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

    # ==========================================
    # 歷史紀錄整合函式
    # ==========================================
    def send_history_reports(self, reply_token: str, reports: list):
        """
        [高階函式] 接收 DB Report 物件列表，自動處理格式轉換並發送
        """
        formatted_records = []

        # 預先取得 JSON 中的設定字典
        theme_dict = self.themes.get("status_themes", {})
        disease_dict = self.themes.get("disease_info", {})
        
        for r in reports:
            # 1. 狀態文字與顏色預設值
            status_text = "檢測中"
            color = "#aaaaaa"
            
            if r.cnn_result:
                # 取得狀態與疾病的字串值
                status_str = r.cnn_result.status.name if hasattr(r.cnn_result.status, "name") else str(r.cnn_result.status)
                disease_val = r.cnn_result.disease.value if hasattr(r.cnn_result.disease, "value") else str(r.cnn_result.disease)

                # 從 JSON 抓取顏色 (與診斷卡片共用 THEME_COLOR)
                theme_config = theme_dict.get(status_str, theme_dict.get("default", {}))
                color = theme_config.get("THEME_COLOR", "#aaaaaa")

                if status_str == "NOT_DETECTED":
                    status_text = "低風險"
                else:
                    # 從 JSON 動態抓取疾病中文名稱
                    disease_config = disease_dict.get(disease_val, disease_dict.get("default", {}))
                    disease_name = disease_config.get("name", "異常")
                    status_text = f"疑似{disease_name}"

            # 2. 時間格式化
            try:
                dt = datetime.fromtimestamp(r.timestamp)
                date_str = dt.strftime("%Y/%m/%d")
            except:
                date_str = str(r.timestamp)

            # 3. 組合 UI 資料
            formatted_records.append({
                "id": r.report_id,
                "date": date_str,
                "status": status_text,
                "color": color
            })

        # 4. 呼叫底層渲染 (如果沒有資料，底層會處理 empty message)
        self.send_history_list(reply_token, formatted_records)

    def send_question(self, reply_token: str, question_data: dict, survey_id: str = None):
        """
        發送問卷題目 (支援 options 格式)
        :param survey_id: 目前問卷的 ID (如 'cataract')，必須由外部傳入
        """
        try:
            text = question_data.get("text", "請回答以下問題")
            
            # 1. 取得選項
            options = question_data.get("options", [])           
            
            quick_reply_buttons = []
            
            # 2. 遍歷 options 產生按鈕
            for opt in options:
                label = opt.get("label")
                value = opt.get("value")
                
                # 取得該選項對應的下一題 ID (支援分支)
                # 若選項內沒寫 next，可退回使用題目層級的 next (若有)
                next_q = opt.get("next") or question_data.get("next")
                
                # 3. 組合 Postback Data (result自動附加 action=submit_survey)
                data = f"value={value}&survey={survey_id}&next={next_q}"
                if next_q == "result":
                    data += "&action=submit_survey"
                
                action = PostbackAction(
                    label=label,
                    data=data,
                    display_text=label
                )
                quick_reply_buttons.append(QuickReplyButton(action=action))

            # 4. 發送訊息
            if quick_reply_buttons:
                message = TextSendMessage(
                    text=text,
                    quick_reply=QuickReply(items=quick_reply_buttons)
                )
                self.api.reply_message(reply_token, message)
            else:
                # 無按鈕時只傳文字
                self.reply_text(reply_token, text)
                
        except Exception as e:
            print(f"❌ Send question failed: {e}")
            self.reply_text(reply_token, "題目載入失敗。")

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