from dotenv import load_dotenv
# 先嘗試載入本地的 .env 檔案；Cloud Run 時靜默忽略 
load_dotenv()
import random
from contextlib import asynccontextmanager
from urllib.parse import quote_plus
from pathlib import Path
from datetime import datetime
import json
import copy
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles

from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, PostbackEvent, FlexSendMessage, FollowEvent
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
user_rag_state = {} # 記錄「衛教諮詢 (RAG)」

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
# 掛載靜態檔案目錄
app.mount("/static", StaticFiles(directory="assets/static"), name="static")
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
        msg = "請傳送「單一」眼睛照片，並確保對焦正確不模糊📸"
        line_service.reply_text(event.reply_token, msg)
        return

    # 3. [歷史紀錄]
    if text in ["歷史紀錄", "查詢紀錄", "History"]:
        try:
            # 讀取樣板
            bubble_container = line_service._load_template("history_list.json")
            row_template = line_service._load_template("history_row.json")

            # C. 從 DB 撈取資料
            reports = db_service.get_reports_by_user(user_id, limit=5)
            
            # 取得容器中用來放資料的 contents 陣列
            # 根據您的 json，位置在 body -> contents
            container_contents = bubble_container["body"]["contents"]

            if not reports:
                # D-1. 如果沒有資料將 placeholder 替換成提示文字
                # 假設 contents[0] 就是 placeholder text component
                json_str = json.dumps(container_contents)
                json_str = json_str.replace("PLACEHOLDER_EMPTY_MSG", "您目前還沒有檢測紀錄喔！")
                bubble_container["body"]["contents"] = json.loads(json_str)
            else:
                # D-2. 如果有資料
                # 1. 先清空容器內的 placeholder (清空原本的 "PLACEHOLDER_EMPTY_MSG" 文字元件)
                container_contents.clear()

                # 2. 遍歷資料並產生 Row
                for r in reports:
                    # --- 邏輯處理  ---
                    status_text = "檢測中"
                    color = "#aaaaaa"
                    
                    if r.cnn_result:
                        if r.cnn_result.status == DiagnosisStatus.NOT_DETECTED:
                            status_text = "低風險"
                            color = "#1DB446"
                        else:
                            disease_map = {"Cataract": "白內障", "Conjunctivitis": "結膜炎", "None": "低風險"}
                            disease_enum_val = r.cnn_result.disease.value if hasattr(r.cnn_result.disease, "value") else str(r.cnn_result.disease)
                            disease_name = disease_map.get(disease_enum_val, disease_enum_val)
                            status_text = f"疑似{disease_name}"
                            if "白內障" in status_text:
                                color = "#EF6C00"
                            elif "結膜炎" in status_text:
                                color = "#D32F2F"
                    
                    try:
                        dt_obj = datetime.fromtimestamp(r.timestamp)
                        date_str = dt_obj.strftime("%Y/%m/%d")
                    except:
                        date_str = str(r.timestamp)

                    # --- 動態生成 UI ---
                    # 1. 深度複製一份 Row 的結構
                    current_row = copy.deepcopy(row_template)
                    # 2. 將 Dict 轉字串以便進行 replace
                    row_str = json.dumps(current_row)
                    # 3. 執行替換
                    row_str = row_str.replace("PLACEHOLDER_DATE", date_str)
                    row_str = row_str.replace("PLACEHOLDER_STATUS", status_text)
                    row_str = row_str.replace("PLACEHOLDER_COLOR", color)
                    row_str = row_str.replace("PLACEHOLDER_REPORT_ID", str(r.report_id))
                    
                    # 4. 轉回 Dict 並加入容器
                    final_row = json.loads(row_str)
                    container_contents.append(final_row)
                    
                    # 入分隔線 separator，讓列表更清楚
                    container_contents.append({"type": "separator", "margin": "md"})

            # E. 發送訊息
            line_service.api.reply_message(
                event.reply_token,
                FlexSendMessage(alt_text="您的歷史檢查紀錄", contents=bubble_container)
            )
            
        except Exception as e:
            logger.error(f"查詢歷史失敗 (JSON Template): {e}")
            line_service.reply_text(event.reply_token, "目前無法讀取紀錄，請稍後再試。")
        return

    # 4. [附近診所]
    if text == "附近診所":
        # 1. 取得 LIFF ID
        liff_id = getattr(settings, "LIFF_ID", None)
        if not liff_id:
            line_service.reply_text(event.reply_token, "系統設定錯誤：找不到 LIFF ID。")
            return

        liff_url = f"https://liff.line.me/{liff_id}"
        
        # 2. 讀取並替換 JSON
        try:
            # 載入剛剛建立的 json 檔
            bubble = line_service._load_template("location_guide.json")
            
            # 將 JSON 轉字串 -> 替換網址 -> 轉回物件
            json_str = json.dumps(bubble)
            json_str = json_str.replace("PLACEHOLDER_LIFF_URL", liff_url)
            final_bubble = json.loads(json_str)
            
            # 3. 發送
            line_service.api.reply_message(
                event.reply_token,
                FlexSendMessage(alt_text="請開啟定位搜尋附近診所", contents=final_bubble)
            )
        except Exception as e:
            logger.error(f"載入定位引導樣板失敗: {e}")
            # 萬一 JSON 讀取失敗，至少回傳個純文字連結當備案
            line_service.reply_text(event.reply_token, f"請點擊連結開啟定位：\n{liff_url}")
            
        return

    # 5. [衛教資訊]
    if text in ["衛教資訊", "更多衛教"]:
        try:
            bubble = line_service._load_template("health_education_menu.json")
            line_service.api.reply_message(
                event.reply_token,
                FlexSendMessage(alt_text="眼科衛教資訊選單", contents=bubble)
            )
        except Exception as e:
            logger.error(f"衛教選單載入失敗: {e}")
            line_service.reply_text(event.reply_token, "暫時無法載入衛教資訊。")
        return

    # 6. [症狀問答] (啟動文字問診流程)
    if text == "症狀問答":
        # 設定問卷檔案與 ID
        survey_filename = "text_mode.json"
        survey_id = "text_mode"

        try:
            # 1. 讀取問卷 JSON
            survey_data = line_service._load_json(Path(f"assets/questionnaires/{survey_filename}"))
            
            if not survey_data:
                logger.error(f"找不到問卷檔案: {survey_filename}")
                line_service.reply_text(event.reply_token, "系統維護中，暫無法載入問卷。")
                return

            # 2. 初始化使用者狀態 (清空過去的回答)
            user_survey_state[user_id] = {
                "current_survey": survey_id,
                "answers": []
            }

            # 3. 發送第一題 (Q1)
            questions = survey_data.get("questions", {})
            
            # 用 Key 取得第一題 (優先讀取 json 裡的 start_question 設定，預設 Q1)
            start_q_id = survey_data.get("start_question", "Q1")
            first_q = questions.get(start_q_id)
            
            if first_q:
                # 必須傳入 survey_id
                line_service.send_question(event.reply_token, first_q, survey_id=survey_id)
            else:
                line_service.reply_text(event.reply_token, "問卷格式錯誤 (找不到 Q1)。")

        except Exception as e:
            logger.error(f"症狀問答啟動失敗: {e}")
            line_service.reply_text(event.reply_token, "發生錯誤，請稍後再試。")
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
            questions = survey_data.get("questions", {})
            start_q_id = survey_data.get("start_question", "Q1")
            first_q = questions.get(start_q_id)
            
            if first_q:
                line_service.send_question(event.reply_token, first_q, survey_id=survey_id)
            else:
                line_service.reply_text(event.reply_token, "問卷格式錯誤 (找不到 Q1)。")

        except Exception as e:
            logger.error(f"啟動問卷失敗: {e}")
            line_service.reply_text(event.reply_token, "啟動失敗，請稍後再試。")
        return
    
    # === 文字問診模式啟動 ===
    if text == "文字問診模式":
        survey_filename = "text_mode.json"
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
            start_q_id = survey_data.get("start_question", "Q1")
            first_q = questions.get(start_q_id)
            if first_q:
                line_service.send_question(event.reply_token, first_q, survey_id=survey_id)
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
                        status_text = "低風險"
                        color = "#1DB446"  # 綠色
                    else:
                        # 顯示病症名稱 (例如: 疑似白內障)
                        disease_map = {
                            "Cataract": "白內障",
                            "Conjunctivitis": "結膜炎",
                            "None": "低風險"
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
    
    # === RAG 衛教問答專用區塊 ===
    if user_rag_state.get(user_id) == True:
        try:
            # 1. 清除狀態
            del user_rag_state[user_id]

            # 2. 載入 RAG 資料庫 (此處保持不變)
            rag_file_path = Path("assets/knowledge/rag_corpus.json")
            context_text = "無相關資料庫內容" # 給預設值，避免 context 為空時 LLM 困惑
            
            if rag_file_path.exists():
                rag_data = line_service._load_json(rag_file_path)
                found_items = []
                # 簡單關鍵字搜尋
                for topic, content in rag_data.items():
                    if topic in text or text in content or any(k in text for k in topic):
                        found_items.append(content)
                
                if found_items:
                    context_text = "\n".join(found_items[:3])

            # 3. 組合 Prompt 
            current_persona = user_personas.get(user_id, "doctor")
            
            # 並將變數透過參數傳入 json key: "rag_consultation"
            final_prompt = llm_service.get_task_prompt(
                "rag_consultation",
                context=context_text,
                question=text,
                persona=current_persona
            )

            # 4. 呼叫 LLM
            reply = llm_service.generate_response(final_prompt, persona=current_persona)
            line_service.reply_text(event.reply_token, reply)
            
        except Exception as e:
            logger.error(f"RAG 流程失敗: {e}")
            line_service.reply_text(event.reply_token, "衛教諮詢發生錯誤，請稍後再試。")
        
        return

    # --- 3. 非指令的文字處理 (Default Fallback) ---
    try:
        fallback_path = Path("assets/fallback_messages.json")
        reply_text = "抱歉，我不太理解您的意思。\n請使用下方選單功能操作。" # 預設

        if fallback_path.exists():
            data = line_service._load_json(fallback_path)
            messages = data.get("messages", [])
            if messages:
                reply_text = random.choice(messages)
        
        line_service.reply_text(event.reply_token, reply_text)

    except Exception as e:
        logger.error(f"讀取訊息失敗: {e}")
        line_service.reply_text(event.reply_token, "請使用下方選單功能。")

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
            line_service.reply_text(event.reply_token, "未能辨認眼睛特徵，請重新對焦與裁切，或調整光線後再試一次。")

    except Exception as e:
        logger.error(f"圖片處理失敗: {e}")
        line_service.reply_text(event.reply_token, "抱歉，圖片分析時發生錯誤，請稍後再試。")

# (C) 處理按鈕回傳 (觸發 CNN)
@handler.add(PostbackEvent)
def handle_postback(event):
    user_id = event.source.user_id
    data = event.postback.data
    logger.info(f"收到 Postback [{user_id}]: {data}")

    try:
        params = dict(x.split('=') for x in data.split('&'))
    except Exception as e:
        # data 可能不是 key=value 格式 (例如 "menu")
        params = {}
        if data == "menu":
             # 簡單處理 menu
             try:
                bubble = line_service._load_template("health_education_menu.json")
                line_service.api.reply_message(event.reply_token, FlexSendMessage(alt_text="選單", contents=bubble))
             except: pass
             return

    action = params.get("action")

    # =================================================
    # 1. 記錄問卷答案 (若有 survey & value)
    # =================================================
    if "survey" in params and "value" in params:
        survey_id = params.get("survey")
        # 過濾掉控制參數
        answer_data = {k: v for k, v in params.items() if k not in ["survey", "next", "action"]}
        
        if user_id not in user_survey_state:
             user_survey_state[user_id] = {"current_survey": survey_id, "answers": []}
        
        user_survey_state[user_id]["answers"].append(answer_data)

    # =================================================
    # 2. Action 分流處理
    # =================================================

    # (A) 問卷提交 -> 產生 LLM 報告
    if action == "submit_survey":
        survey_id = params.get("survey")
        try:
            answers = user_survey_state.get(user_id, {}).get("answers", [])
            answers_str = "\n".join([f"- {a}" for a in answers])
            
            prompt = llm_service.get_task_prompt(
                "questionnaire_summary", 
                survey_id=survey_id, 
                answers_str=answers_str
            )
            
            current_persona = user_personas.get(user_id, "doctor")
            reply = llm_service.generate_response(prompt, persona=current_persona)
            line_service.reply_text(event.reply_token, reply)
            
        except Exception as e:
            logger.error(f"問卷報告產生失敗: {e}")
            line_service.reply_text(event.reply_token, "產生報告時發生錯誤，但您的回答紀錄已保存。")
        
        # 清除狀態
        if user_id in user_survey_state:
            del user_survey_state[user_id]
        return

    # (B) 啟動 RAG 衛教諮詢
    elif action == "ask_llm":
        user_rag_state[user_id] = True 
        msg = "請輸入您想詢問的衛教內容 (10 字內) 📝\n\n例如：「白內障術後保養」"
        line_service.reply_text(event.reply_token, msg)
        return

    # (C) 圖片診斷確認 (CNN)
    elif action == "confirm_cnn":
        report_id = params.get("report_id")
        if report_id:
            try:
                report = db_service.get_report(report_id)
                if not report:
                    line_service.reply_text(event.reply_token, "找不到此診斷紀錄。")
                    return
                
                final_report = image_service.run_cnn_phase(report)
                db_service.save_report(final_report)
                line_service.send_analysis_result(event.reply_token, final_report)

            except Exception as e:
                logger.error(f"CNN 分析失敗: {e}")
                line_service.reply_text(event.reply_token, "分析過程中發生錯誤。")
        return

    # (D) 查看歷史報告
    elif action == "view_report":
        report_id = params.get("report_id")
        if report_id:
            try:
                report = db_service.get_report(report_id)
                if report:
                    line_service.send_analysis_result(event.reply_token, report)
                else:
                    line_service.reply_text(event.reply_token, "找不到該筆報告資料。")
            except: pass
        return

    # (E) 重新檢測
    elif action == "retry":
        line_service.reply_text(event.reply_token, "請重新上傳一張清楚的眼睛照片。")
        return

    # (F) 問卷下一題 (若沒有命中 submit_survey 但有 next)
    elif "survey" in params and "next" in params:
        survey_id = params.get("survey")
        next_q_id = params.get("next")
        
        try:
            filename = f"{survey_id}.json"
            survey_data = line_service._load_json(Path(f"assets/questionnaires/{filename}"))
            questions = survey_data.get("questions", {})
            next_q = questions.get(next_q_id)
            
            if next_q:
                line_service.send_question(event.reply_token, next_q, survey_id=survey_id)
            else:
                line_service.reply_text(event.reply_token, "系統錯誤：找不到下一題。")
        except Exception as e:
            logger.error(f"問卷切換失敗: {e}")
        return
    
    # (g) 顯示衛教詳情 
    if action == "view_education":
        topic = params.get("topic")
        
        # 建立 Topic 與 JSON 檔名的對照表
        template_map = {
            "cataract": "education_cataract.json",
            "conjunctivitis": "education_conjunctivitis.json",
            "prevention": "education_prevention.json",
            "白內障": "education_cataract.json",
            "結膜炎": "education_conjunctivitis.json"
        }
        
        # 取得對應的檔名
        filename = template_map.get(topic)
        
        if filename:
            try:
                # 載入對應的 JSON 樣板
                bubble = line_service._load_template(filename)
                
                # 根據 topic 設定 alt_text (推播通知預覽文字)
                alt_text_map = {
                    "cataract": "認識白內障",
                    "conjunctivitis": "認識結膜炎",
                    "prevention": "日常預防保健"
                }
                alt_text = alt_text_map.get(topic, "衛教資訊")

                line_service.api.reply_message(
                    event.reply_token,
                    FlexSendMessage(alt_text=alt_text, contents=bubble)
                )
            except Exception as e:
                logger.error(f"衛教詳情載入失敗 ({topic}): {e}")
                line_service.reply_text(event.reply_token, "暫時無法載入該衛教資訊。")
        else:
            line_service.reply_text(event.reply_token, "找不到此衛教主題。")
        
        return

    # 其他未處理 Action
    else:
        logger.debug(f"未處理的 Postback: {params}")

# (D) 處理加入好友事件 (發送 Welcome Card)
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