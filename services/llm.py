import json
import os
from pathlib import Path
from typing import Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from config import settings
from services.log import logger

class LLMService:
    def __init__(self):
        # 1. 初始化 Client (支援 Groq 與 OpenAI)
        api_key = settings.OPENAI_API_KEY
        base_url = settings.OPENAI_BASE_URL
        
        if OpenAI and api_key:
            logger.info(f"🧠 LLM Service init with model: {settings.LLM_MODEL}")
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            self.enabled = True
        else:
            self.client = None
            self.enabled = False
            logger.warning("⚠️ LLM 未啟用，將使用模擬模式。")

        # 2. 載入靜態資源
        self.prompts_path = Path("assets/prompts/system_prompts.json")
        self.corpus_path = Path("assets/knowledge/rag_corpus.json")
        
        self.system_prompts = self._load_json(self.prompts_path)
        self.rag_corpus = self._load_json(self.corpus_path)

    def _load_json(self, path: Path) -> Dict:
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"讀取 {path} 失敗: {e}")
            return {}

    def get_system_prompt(self, persona: str = "doctor") -> str:
        base = self.system_prompts.get("common_rules", "你是一個眼科助手。")
        roles = self.system_prompts.get("roles", {})
        role_data = roles.get(persona, roles.get("doctor", {}))
        return f"{base}\n\n{role_data.get('prompt', '')}"
    
    def get_task_prompt(self, task_name: str, **kwargs) -> str:
        """
        取得任務型 Prompt 並填入變數
        使用方式: get_task_prompt("questionnaire_summary", survey_id="白內障", answers_str="...")
        """
        tasks = self.system_prompts.get("tasks", {})
        raw_prompt = tasks.get(task_name, "")
        
        # 自動替換變數 (如 {survey_id})
        try:
            return raw_prompt.format(**kwargs)
        except KeyError as e:
            logger.error(f"Prompt 變數缺失: {e}")
            return raw_prompt # 回傳未替換的字串以免報錯

    def get_knowledge_context(self, keyword: str) -> str:
        """簡易 RAG 檢索"""
        if not keyword: return ""
        hits = [v for k, v in self.rag_corpus.items() if keyword in k or keyword in v[:20]]
        return "【參考醫學資料】\n" + "\n".join(hits[:2]) if hits else ""

    def generate_response(self, user_text: str, persona: str = "doctor", context_keyword: Optional[str] = None) -> str:
        if not self.enabled:
            return f"[系統模擬 ({settings.LLM_MODEL})]: {user_text}"

        try:
            system_msg = self.get_system_prompt(persona)
            if context_keyword:
                system_msg += f"\n\n{self.get_knowledge_context(context_keyword)}"

            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL, # 使用 Config 中的模型名稱
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return "抱歉，AI 服務暫時無法連線。"

llm_service = LLMService()