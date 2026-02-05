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

    #  --- 從 DB 讀取使用者狀態 ---
    user_state = db_service.get_user_state(user_id)
    
    # 取得當前風格 (預設 doctor)
    current_persona = user_state.get("persona", "doctor")
    # 取得 RAG 狀態
    is_rag_mode = user_state.get("rag_mode", False)
    rag_topic = user_state.get("rag_topic", None)

    # === 問卷啟動指令 ===
    if text in ["白內障檢測", "結膜炎檢測", "文字問診模式", "症狀問答"]:
        # 1. 決定 survey_id 與 filename
        if text == "白內障檢測":
            filename, s_id = "cataract.json", "cataract"
        elif text == "結膜炎檢測":
            filename, s_id = "conjunctivitis.json", "conjunctivitis"
        else:
            filename, s_id = "text_mode.json", "text_mode"

        try:
            survey_data = line_service._load_json(Path(f"assets/questionnaires/{filename}"))
            if not survey_data:
                line_service.reply_text(event.reply_token, "找不到問卷檔案。")
                return

            # 初始化 DB 中的問卷狀態 (清空 answers)
            db_service.update_survey_progress(user_id, s_id, [])

            # 發送第一題
            questions = survey_data.get("questions", {})
            start_q_id = survey_data.get("start_question", "Q1")
            first_q = questions.get(start_q_id)
            
            if first_q:
                line_service.send_question(event.reply_token, first_q, survey_id=s_id)
            else:
                line_service.reply_text(event.reply_token, "問卷格式錯誤。")
        except Exception as e:
            logger.error(f"問卷啟動失敗: {e}")
            line_service.reply_text(event.reply_token, "啟動失敗，請稍後再試。")
        return
    
    # === RAG 衛教問答 ===
    if is_rag_mode:

        if len(text) > 25:
            msg = "您的問題太長囉！請精簡至「20字內」重新輸入，或輸入「取消」來結束問答。"
            
            # 判斷是否要手動退出
            if text == "取消":
                db_service.update_rag_mode(user_id, False)
                line_service.reply_text(event.reply_token, "已結束衛教諮詢。")
                return

            line_service.reply_text(event.reply_token, msg)
            return
        
        try:
            # 問完一次後自動關閉，避免使用者卡在衛教模式
            db_service.update_rag_mode(user_id, False)

            # 2. 載入 RAG 資料庫
            rag_file_path = Path("assets/knowledge/rag_corpus.json")
            context_text = "無相關資料庫內容"
            
            if rag_file_path.exists():
                rag_data = line_service._load_json(rag_file_path)
                found_items = []
                for topic, content in rag_data.items():
                    if topic in text or text in content:
                        found_items.append(content)
                if found_items:
                    context_text = "\n".join(found_items[:3])

            # 3. 呼叫 LLM (加入 topic 資訊若有)
            prompt_suffix = f"(Focus on {rag_topic})" if rag_topic else ""
            final_prompt = llm_service.get_task_prompt(
                "rag_consultation",
                context=context_text,
                question=text + prompt_suffix,
                persona=current_persona
            )

            reply = llm_service.generate_response(final_prompt, persona=current_persona)
            line_service.reply_text(event.reply_token, reply)
            
        except Exception as e:
            logger.error(f"RAG 流程失敗: {e}")
            line_service.reply_text(event.reply_token, "衛教諮詢發生錯誤。")
        return
    
    # --- 3. 非指令的文字處理 (Default Fallback) ---
    try:
        fallback_path = Path("assets/static/fallback_messages.json")
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
        params = dict(x.split('=', 1) for x in data.split('&'))
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

    # ==========================================
    # 從 DB 讀取狀態
    # ==========================================
    user_state = db_service.get_user_state(user_id)
    current_persona = user_state.get("persona", "doctor")
    survey_state = user_state.get("survey", {}) # 取得問卷狀態
    if survey_state is None: survey_state = {}

    # =================================================
    # 1. 記錄問卷答案 (若有 survey & value)
    # =================================================
    if "survey" in params and "value" in params:
        survey_id = params.get("survey")
        answer_data = {k: v for k, v in params.items() if k not in ["survey", "next", "action"]}
        
        # 讀取舊答案 -> 加入新答案 -> 存回 DB
        existing_answers = survey_state.get("answers", [])
        existing_answers.append(answer_data)
        
        db_service.update_survey_progress(user_id, survey_id, existing_answers)

    # =================================================
    # 2. Action 分流處理
    # =================================================

    # 1. [症狀問答] (對應: symptom_qa)
    if action == "symptom_qa":
        filename, s_id = "text_mode.json", "text_mode"
        try:
            survey_data = line_service._load_json(Path(f"assets/questionnaires/{filename}"))
            if survey_data:
                db_service.update_survey_progress(user_id, s_id, []) # 初始化
                questions = survey_data.get("questions", {})
                start_q_id = survey_data.get("start_question", "Q1")
                first_q = questions.get(start_q_id)
                if first_q:
                    line_service.send_question(event.reply_token, first_q, survey_id=s_id)
        except Exception as e:
            logger.error(f"Postback 問卷啟動失敗: {e}")
        return
    
    # 2. [開始檢測] (對應: start_test)
    elif action == "start_test":
        line_service.send_camera_request(event.reply_token)
        return
    
    # 3. [歷史紀錄] (對應: history)
    elif action == "history":
        try:
            # 1. 從 DB 撈取該使用者的紀錄
            reports = db_service.get_reports_by_user(user_id, limit=5)
            
            # 2. 直接呼叫 line.py 中封裝好的函式
            # (它會自動處理日期格式、狀態判斷、顏色設定並發送 Flex Message)
            line_service.send_history_reports(event.reply_token, reports)
            
        except Exception as e:
            logger.error(f"Postback 查詢歷史失敗: {e}")
            line_service.reply_text(event.reply_token, "系統忙碌中，無法讀取紀錄。")
        return
    
    # 4. [衛教資訊] (對應: education)
    elif action == "education":
        try:
            bubble = line_service._load_template("health_education_menu.json")
            line_service.api.reply_message(event.reply_token, FlexSendMessage(alt_text="衛教資訊選單", contents=bubble))
        except Exception as e:
            logger.error(f"Postback 衛教選單失敗: {e}")
        return
    
    # 5. [風格設定] (對應: style_setting)
    elif action == "style_setting":
        try:
            bubble = line_service._load_template("type_selection.json")
            line_service.api.reply_message(event.reply_token, FlexSendMessage(alt_text="請選擇助手風格", contents=bubble))
        except:
            line_service.reply_text(event.reply_token, "無法載入風格選單。")
        return
    
    # -----------------------------------------------------------
    #  6. 處理風格切換 (接收選單回傳的動作)
    # -----------------------------------------------------------
    elif action == "set_style":
        selected_role = params.get("mode")
        
        # 1. 定義角色對照表 
        role_map = {
            "doctor": "資深診療師",
            "nurse": "溫柔護理師",
            "comedian": "喜劇漫才師",
            "parent": "人生說教家",
            "angel": "護眼小天使",
            "engineer": "軟體工程師"
        }

        # 2. 角色專屬台詞表，定義每個角色切換成功後要說的一句話
        role_flavor_text = {
            "doctor": "我會以專業醫學數據與臨床經驗為您分析。",
            "nurse": "別擔心，深呼吸～讓我來溫柔地協助您檢查。",
            "comedian": "眼睛跟笑話一樣，都要有『亮點』才行！嘿嘿！",
            "parent": "我是為你好！整天看手機，還不快去檢查！", 
            "angel": "我會用愛與魔法守護您的靈魂之窗喔！✨",
            "engineer": "System initialized. Logic module loaded. 0 errors found."
        }

        # 3. 檢查是否為有效角色 (直接比對，不轉小寫)
        if selected_role and selected_role in role_map:
            
            # 更新 DB
            db_service.update_persona(user_id, selected_role)

            # 準備回覆文字
            display_name = role_map.get(selected_role)
            flavor_msg = role_flavor_text.get(selected_role, "")

            reply_text = f"已切換為【{display_name}】風格！\n{flavor_msg}"

            line_service.reply_text(event.reply_token, reply_text)
        else:
            logger.warning(f"無效的風格請求: {selected_role}")
            line_service.reply_text(event.reply_token, "找不到此風格設定。")

        return
    
    # 7. [關於我們] (對應: welcome_msg)
    elif action == "welcome_msg":
        try:
            bubble = line_service._load_template("welcome.json")
            line_service.api.reply_message(event.reply_token, FlexSendMessage(alt_text="關於我們", contents=bubble))
        except:
            pass
        return

    # (A) 問卷提交 -> 產生 LLM 報告
    if action == "submit_survey":
        survey_id = params.get("survey")
        try:
            answers = survey_state.get(user_id, {}).get("answers", [])
            answers_str = "\n".join([f"- {a}" for a in answers])
            
            prompt = llm_service.get_task_prompt(
                "questionnaire_summary", 
                survey_id=survey_id, 
                answers_str=answers_str
            )
            
            reply = llm_service.generate_response(prompt, persona=current_persona)
            line_service.reply_text(event.reply_token, reply)
            
        except Exception as e:
            logger.error(f"問卷報告產生失敗: {e}")
            line_service.reply_text(event.reply_token, "產生報告時發生錯誤，但您的回答紀錄已保存。")
        
        # 清除狀態
        db_service.clear_survey(user_id)
        return

    # (B) 啟動 RAG 衛教諮詢
    elif action == "ask_llm":
        topic = params.get("topic")
        # 設定 RAG 模式為 True
        db_service.update_rag_mode(user_id, True, topic)
        msg = "請輸入您想詢問的衛教內容 (20 字內) 📝"
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