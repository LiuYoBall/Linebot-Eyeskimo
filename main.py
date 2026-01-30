from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
import json
from fastapi import FastAPI, Request, HTTPException

from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, PostbackEvent, 
    FlexSendMessage, LocationMessage, FollowEvent
)

# 匯入模組
from config import settings
from services import (
    logger, image_service, line_service, 
    db_service, llm_service
)
from models import ai_manager
from schemas import DiagnosisStatus

# 全域變數宣告 (Cloud Run多開環境下會不穩定，建議未來遷移至 Firestore)
user_personas = {}
user_survey_state = {} # 記憶問卷答案 

# ==========================================
# 1. 生命週期管理 (啟動/關閉)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Server starting... Warming up AI Models...")
    try:
        _ = ai_manager.yolo
        _ = ai_manager.cnn
        logger.info("✅ AI Models ready! System is online.")
    except Exception as e:
        logger.error(f"❌ AI Models init failed: {e}")
    yield
    logger.info("🛑 Server shutting down...")

app = FastAPI(lifespan=lifespan)
handler = line_service.handler

# ==========================================
# 2. API 路由
# ==========================================
@app.get("/")
def health_check():
    """健康檢查端點 (給 Cloud Run 偵測用)"""
    return {"status": "ok", "version": "1.0.0"}

@app.post("/callback")
async def callback(request: Request):
    """LINE Webhook 入口"""
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    body_str = body.decode("utf-8")

    try:
        handler.handle(body_str, signature)
    except InvalidSignatureError:
        logger.error("Invalid Signature")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Webhook Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return "OK"

# ==========================================
# 3. LINE 事件處理邏輯
# ==========================================

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()
    
    logger.info(f"收到文字 [{user_id}]: {text}")

    # --- Rich Menu 按鈕處理 ---

    # 1. [風格設定]
    if text == "風格設定":
        try:
            bubble = line_service._load_template("type_selection.json")
            line_service.api.reply_message(
                event.reply_token,
                FlexSendMessage(alt_text="請選擇助手風格", contents=bubble)
            )
        except Exception as e:
            logger.error(f"風格選單載入失敗: {e}")
            line_service.reply_text(event.reply_token, "暫時無法載入風格選單。")
        return
    
    # 2. [開始檢測]
    if text == "開始檢測":
        # 引導使用者上傳圖片或選擇文字模式
        msg = "請直接傳送一張「眼睛照片」給我進行分析 📸\n\n或者輸入「白內障檢測」/「結膜炎檢測」使用文字問卷模式。"
        line_service.reply_text(event.reply_token, msg)
        return

    # 3. [歷史紀錄]
    if text in ["歷史紀錄", "查詢紀錄", "History"]:
        try:
            reports = db_service.get_reports_by_user(user_id, limit=5)
            history_data = []
            for r in reports:
                status_text = "檢測中"
                color = "#aaaaaa"
                if r.cnn_result:
                    if r.cnn_result.status == DiagnosisStatus.NOT_DETECTED:
                        status_text = "正常 / 低風險"
                        color = "#1DB446"
                    else:
                        disease_map = {"Cataract": "白內障", "Conjunctivitis": "結膜炎", "None": "正常"}
                        disease_enum_val = r.cnn_result.disease.value if hasattr(r.cnn_result.disease, "value") else str(r.cnn_result.disease)
                        disease_name = disease_map.get(disease_enum_val, disease_enum_val)
                        status_text = f"疑似{disease_name}"
                        color = "#D32F2F" if "結膜炎" in status_text else "#EF6C00"
                
                try:
                    dt_obj = datetime.fromtimestamp(r.timestamp)
                    date_str = dt_obj.strftime("%Y/%m/%d")
                except:
                    date_str = str(r.timestamp)

                history_data.append({"id": r.report_id, "date": date_str, "status": status_text, "color": color})
            
            line_service.send_history_list(event.reply_token, history_data)
        except Exception as e:
            logger.error(f"查詢歷史失敗: {e}")
            line_service.reply_text(event.reply_token, "目前無法讀取紀錄，請稍後再試。")
        return

    # 4. [附近診所]
    if text == "附近診所":
        line_service.reply_text(event.reply_token, "請點擊對話框左下的「+」號，選擇「位置資訊」並傳送您的位置，我將為您搜尋附近的眼科診所 🏥")
        return

    # 5. [衛教資訊]
    if text == "衛教資訊":
        # 這裡可以回傳一個簡單的選單或文字
        # 假設您之後會做一個 health_info.json，目前先用文字回應
        msg = "【常見眼疾衛教】\n\n👁️ 白內障：水晶體混濁，造成視力模糊。\n👁️ 結膜炎：眼睛發紅、分泌物增加。\n\n請保持用眼衛生，定期檢查！"
        line_service.reply_text(event.reply_token, msg)
        return

    # 6. [症狀問答] (引導進入 LLM 模式)
    if text == "症狀問答":
        current_persona = user_personas.get(user_id, "doctor")
        # 這裡不直接回傳，而是讓使用者知道可以開始問
        line_service.reply_text(event.reply_token, "請告訴我您目前的眼睛狀況，我將為您提供初步建議。(例如：眼睛紅紅的、覺得看東西模糊...)")
        return

    # --- 2. 處理風格切換指令 ---
    if text.startswith("切換風格："):
        # 取出冒號後面的英文代碼 (e.g., doctor, nurse...)
        selected_role = text.split("：")[1].strip()
        # 驗證是否為有效角色 (防呆)
        valid_roles = llm_service.system_prompts.get("roles", {}).keys()
        
        if selected_role in valid_roles:
            user_personas[user_id] = selected_role # 記錄   
            # 給予對應回覆
            role_names = {
                "doctor": "專業醫師",
                "nurse": "溫柔護理師",
                "comedian": "幽默演員",
                "asian_parent": "亞洲父母"
            }
            role_name = role_names.get(selected_role, selected_role)
            line_service.reply_text(event.reply_token, f"已切換為【{role_name}】風格！請把照片傳給我吧！")
        else:
            line_service.reply_text(event.reply_token, "無效的角色選擇。")
        return
    
    # === 問卷啟動指令 ===
    # 當使用者輸入 "白內障檢測" 或 "結膜炎檢測" 時觸發
    if text in ["白內障檢測", "結膜炎檢測"]:
        # 1. 決定要讀哪份問卷
        survey_filename = "cataract.json" if text == "白內障檢測" else "conjunctivitis.json"
        survey_id = survey_filename.replace(".json", "") # 取得 ID (如 cataract)

        try:
            # 2. 讀取問卷 JSON
            # 使用 line_service 內部的讀取方法 (或者也可以用 json.load)
            survey_data = line_service._load_json(Path(f"assets/questionnaires/{survey_filename}"))
            
            if not survey_data:
                line_service.reply_text(event.reply_token, "找不到問卷檔案。")
                return

            # 3. 初始化使用者的狀態
            user_survey_state[user_id] = {
                "current_survey": survey_id,
                "answers": []
            }

            # 4. 發送第一題 (通常是 id="Q1")
            first_q = next((q for q in survey_data["questions"] if q["id"] == "Q1"), None)
            
            if first_q:
                line_service.send_question(event.reply_token, first_q)
            else:
                line_service.reply_text(event.reply_token, "問卷格式錯誤 (找不到 Q1)。")

        except Exception as e:
            logger.error(f"啟動問卷失敗: {e}")
            line_service.reply_text(event.reply_token, "啟動失敗，請稍後再試。")
        return
    
    # === 文字問診模式啟動 ===
    if text == "文字問診模式":
        survey_filename = "text_mode_flow.json"
        survey_id = "text_mode"

        try:
            # 讀取共用的文字問診流程
            survey_data = line_service._load_json(Path(f"assets/questionnaires/{survey_filename}"))
            
            if not survey_data:
                line_service.reply_text(event.reply_token, "系統維護中 (找不到問卷檔案)。")
                return

            # 初始化狀態
            user_survey_state[user_id] = {
                "current_survey": survey_id,
                "answers": []
            }

            # 發送第一題
            first_q = next((q for q in survey_data["questions"] if q["id"] == "Q1"), None)
            if first_q:
                line_service.send_question(event.reply_token, first_q)
            else:
                line_service.reply_text(event.reply_token, "問卷啟動失敗。")

        except Exception as e:
            logger.error(f"文字問診啟動失敗: {e}")
            line_service.reply_text(event.reply_token, "發生錯誤，請稍後再試。")
        return
    
    # === 歷史紀錄查詢 ===
    if text in ["查詢紀錄", "歷史紀錄", "History"]:
        try:
            # 1. 從 DB 撈取該使用者的紀錄 (取得 DiagnosticReport 物件列表)
            reports = db_service.get_reports_by_user(user_id, limit=5)
            
            # 2. 轉換資料格式 (DiagnosticReport -> UI Dict)
            history_data = []
            for r in reports:
                # 判斷狀態顏色與顯示文字
                status_text = "檢測中"
                color = "#aaaaaa"
                
                # 使用 DiagnosisStatus Enum 比對 (Issue from screenshots)
                if r.cnn_result:
                    if r.cnn_result.status == DiagnosisStatus.NOT_DETECTED:
                        status_text = "正常 / 低風險"
                        color = "#1DB446"  # 綠色
                    else:
                        # 顯示病症名稱 (例如: 疑似白內障)
                        disease_map = {
                            "Cataract": "白內障",
                            "Conjunctivitis": "結膜炎",
                            "None": "正常"
                        }
                        # 取得英文 enum 值 (str)
                        disease_enum_val = r.cnn_result.disease.value if hasattr(r.cnn_result.disease, "value") else str(r.cnn_result.disease)
                        disease_name = disease_map.get(disease_enum_val, disease_enum_val)

                        status_text = f"疑似{disease_name}"
                        # 根據病症給顏色 (這裡可以簡單用紅色代表異常，或細分)
                        if "白內障" in status_text:
                            color = "#EF6C00" # 橘色
                        elif "結膜炎" in status_text:
                            color = "#D32F2F" # 紅色
                
                # 格式化時間
                try:
                    # 將 int timestamp 轉為 datetime 物件
                    dt_obj = datetime.fromtimestamp(r.timestamp)
                    date_str = dt_obj.strftime("%Y/%m/%d")
                except Exception:
                    # 預防萬一 timestamp 格式有誤
                    date_str = str(r.timestamp)

                history_data.append({
                    "id": r.report_id,
                    "date": date_str,
                    "status": status_text,
                    "color": color
                })
            
            # 3. 發送列表
            line_service.send_history_list(event.reply_token, history_data)
            
        except Exception as e:
            logger.error(f"查詢歷史失敗: {e}")
            line_service.reply_text(event.reply_token, "系統忙碌中，無法讀取紀錄。")
        return
    
    # --- 3. 一般對話 (LLM) ---
    # 預設使用 doctor，若使用者有設定過則用設定的
    current_persona = user_personas.get(user_id, "doctor")
    
    # 產生回應
    reply = llm_service.generate_response(text, persona=current_persona)
    line_service.reply_text(event.reply_token, reply)

# (B) 處理圖片訊息 (觸發 YOLO)
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    user_id = event.source.user_id
    msg_id = event.message.id
    logger.info(f"收到圖片 [{user_id}], ID: {msg_id}")

    try:
        # 1. 下載圖片內容
        message_content = line_service.api.get_message_content(msg_id)
        image_bytes = message_content.content
        
        # 2. 執行 Phase 1 (YOLO)
        report = image_service.run_yolo_phase(user_id, image_bytes)
        
        # 3. 存入資料庫
        db_service.save_report(report)
        
        # 4. 根據結果回應
        if report.yolo_result and report.yolo_result.is_detected:
            # 成功偵測 -> 發送確認卡片
            line_service.send_crop_confirmation(event.reply_token, report)
        else:
            # 未偵測到 -> 提示重拍
            line_service.reply_text(event.reply_token, "未能偵測到清晰的眼睛特徵，請試著靠近一點拍攝，或調整光線後再試一次。")

    except Exception as e:
        logger.error(f"圖片處理失敗: {e}")
        line_service.reply_text(event.reply_token, "抱歉，圖片分析時發生錯誤，請稍後再試。")

# (C) 處理按鈕回傳 (觸發 CNN)
@handler.add(PostbackEvent)
def handle_postback(event):
    user_id = event.source.user_id
    data = event.postback.data
    logger.info(f"收到 Postback [{user_id}]: {data}")

    # 1. 解析參數 (將 "key=value&a=b" 轉為 Dictionary)
    try:
        params = dict(x.split('=') for x in data.split('&'))
    except Exception as e:
        logger.error(f"Postback 參數解析失敗: {data}, Error: {e}")
        return

    action = params.get("action")

    # =================================================
    # 🔀 分支 A: 圖片診斷確認 (Action: confirm_cnn)
    # =================================================
    if action == "confirm_cnn":
        report_id = params.get("report_id")
        
        if report_id:
            try:
                # 1. 從 DB 撈回報告
                report = db_service.get_report(report_id)
                if not report:
                    line_service.reply_text(event.reply_token, "找不到此診斷紀錄，請重新上傳。")
                    return

                # 2. 執行 Phase 2 (CNN)
                final_report = image_service.run_cnn_phase(report)
                
                # 3. 更新 DB
                db_service.save_report(final_report)
                
                # 4. 發送最終結果
                line_service.send_analysis_result(event.reply_token, final_report)

            except Exception as e:
                logger.error(f"CNN 分析失敗: {e}")
                line_service.reply_text(event.reply_token, "分析過程中發生錯誤，請稍後再試。")
        else:
            logger.error("Postback 缺少 report_id")

    # =================================================
    # 🔀 分支 B: 問卷回答 (特徵: 包含 survey 與 next)
    # =================================================
    elif "survey" in params and "next" in params:
        survey_id = params.get("survey")
        next_q_id = params.get("next")
        
        # 1. 記錄答案
        # 過濾掉控制參數 (survey, next)，只留真正有意義的 key/value
        answer_data = {k: v for k, v in params.items() if k not in ["survey", "next"]}
        
        # 確保使用者狀態存在 (使用全域變數 user_survey_state)
        if user_id not in user_survey_state:
             user_survey_state[user_id] = {"current_survey": survey_id, "answers": []}
        
        # 加入這題的答案
        user_survey_state[user_id]["answers"].append(answer_data)
        
        # 2. 判斷下一步
        if next_q_id == "result":
            # === (B-1) 問卷結束 -> 產生 LLM 報告 ===
            try:
                # 取得累積的所有答案
                answers = user_survey_state[user_id]["answers"]
                # 將答案轉為字串給 LLM 看
                answers_str = "\n".join([f"- {a}" for a in answers])
                
                # 使用 get_task_prompt 從 JSON 讀取設定
                prompt = llm_service.get_task_prompt(
                    "questionnaire_summary", 
                    survey_id=survey_id, 
                    answers_str=answers_str
                )
                
                # 呼叫 LLM (使用當前設定的角色，或強制用 doctor)
                current_persona = user_personas.get(user_id, "doctor")
                reply = llm_service.generate_response(prompt, persona=current_persona)
                
                line_service.reply_text(event.reply_token, reply)
                
            except Exception as e:
                logger.error(f"問卷分析失敗: {e}")
                line_service.reply_text(event.reply_token, "產生報告時發生錯誤，但您的回答紀錄已保存。")
            
            # 清除狀態 (重置)
            if user_id in user_survey_state:
                del user_survey_state[user_id]

        else:
            # === (B-2) 繼續下一題 (Next Question) ===
            try:
                # 讀取對應的 JSON 檔
                filename = f"{survey_id}.json"
                survey_data = line_service._load_json(Path(f"assets/questionnaires/{filename}"))
                # 使用 next() 搭配 generator 尋找下一題物件，搜尋 id 符合的題目
                next_q = next((q for q in survey_data.get("questions", []) if q["id"] == next_q_id), None)
                
                if next_q:
                    line_service.send_question(event.reply_token, next_q)
                else:
                    logger.error(f"找不到題目 ID: {next_q_id}")
                    line_service.reply_text(event.reply_token, "系統錯誤：找不到下一題。")
                    
            except Exception as e:
                logger.error(f"問卷切換失敗: {e}")
                line_service.reply_text(event.reply_token, "讀取問卷時發生錯誤。")

    # =================================================
    # 🔀 分支 C: 其他操作 (如 "重新檢測" action=retry)
    # =================================================
    elif action == "retry":
        line_service.reply_text(event.reply_token, "好的，請重新上傳一張清楚的眼睛照片，或輸入「白內障檢測」開始問卷。")
    
        # === 查看歷史報告詳細內容 ===
    elif action == "view_report":
        report_id = params.get("report_id")
        if report_id:
            try:
                # 1. 從 DB 撈取完整報告
                report = db_service.get_report(report_id)
                if report:
                    line_service.send_analysis_result(event.reply_token, report)
                else:
                    line_service.reply_text(event.reply_token, "找不到該筆報告資料 (可能已過期)。")
            except Exception as e:
                logger.error(f"讀取報告失敗: {e}")
        else:
            logger.error("缺少 report_id")
    
    else:
        logger.warning(f"未知的 Postback action: {params}")


# (D) 處理位置訊息 (尋找診所)
@handler.add(MessageEvent, message=LocationMessage)
def handle_location_message(event):
    user_id = event.source.user_id
    lat = event.message.latitude
    lon = event.message.longitude
    address = event.message.address

    logger.info(f"收到位置 [{user_id}]: {address}")

    # 1. 準備資料
    google_map_url = f"https://www.google.com/maps/search/?api=1&query=眼科&query_place_id={lat},{lon}"
    
    # 2. 讀取 JSON 樣板 (使用 line_service 的 helper)
    try:
        bubble = line_service._load_template("location_result.json")
        # 3. 替換變數 (轉字串 -> replace -> 轉回物件)
        json_str = json.dumps(bubble)
        json_str = json_str.replace("PLACEHOLDER_ADDRESS", address)
        json_str = json_str.replace("PLACEHOLDER_URL", google_map_url)
        
        final_bubble = json.loads(json_str)

        # 4. 發送
        line_service.api.reply_message(
            event.reply_token,
            FlexSendMessage(alt_text="附近診所搜尋結果", contents=final_bubble)
        )
    except Exception as e:
        logger.error(f"發送位置結果失敗: {e}")
        line_service.reply_text(event.reply_token, f"搜尋連結：{google_map_url}")

# (E) 處理加入好友事件 (發送 Welcome Card)
@handler.add(FollowEvent)
def handle_follow(event):
    user_id = event.source.user_id
    logger.info(f"新使用者加入: {user_id}")
    try:
        # 讀取 welcome.json
        bubble = line_service._load_template("welcome.json")
        # 傳送歡迎訊息 (如果是 Carousel，contents 就是 bubble 本身)
        line_service.api.reply_message(
            event.reply_token,
            FlexSendMessage(alt_text="歡迎使用 AI 眼科助理", contents=bubble)
        )
    except Exception as e:
        logger.error(f"發送歡迎訊息失敗: {e}")   

# 本地測試用 (當直接執行 main.py 時)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)