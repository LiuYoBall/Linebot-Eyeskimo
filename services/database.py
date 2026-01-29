import firebase_admin
from firebase_admin import credentials, firestore
from config import settings
from schemas import DiagnosticReport

class DatabaseService:
    def __init__(self):
        # 1. 初始化 Firebase Admin SDK
        # 檢查是否已經初始化過 (避免 Hot Reload 時重複初始化報錯)
        if not firebase_admin._apps:
            print(f"🔥 Initializing Firestore for project: {settings.GCP_PROJECT_ID}")
            
            # 在 Cloud Run 或有安裝 gcloud 的本地端，通常不需要手動給 key
            # 它會自動抓取 Application Default Credentials (ADC)
            try:
                firebase_admin.initialize_app(options={
                    'projectId': settings.GCP_PROJECT_ID
                })
            except Exception as e:
                print(f"⚠️ Firebase init failed (若本地開發請確認已登入 gcloud): {e}")

        # 2. 取得 Firestore Client
        self.db = firestore.client()
        self.collection = "diagnostic_reports"

    def save_report(self, report: DiagnosticReport) -> bool:
        """
        儲存或更新診斷報告
        輸入: DiagnosticReport 物件
        """
        try:
            # Pydantic 轉 Dict (exclude_none=False 確保欄位完整)
            report_dict = report.model_dump(mode='json')
            
            # 寫入 Firestore (使用 report_id 當作 Document ID)
            doc_ref = self.db.collection(self.collection).document(report.report_id)
            doc_ref.set(report_dict, merge=True)
            
            print(f"💾 Report saved: {report.report_id} (Status: {report.current_status})")
            return True
        except Exception as e:
            print(f"❌ Failed to save report {report.report_id}: {e}")
            return False

    def get_report(self, report_id: str) -> DiagnosticReport | None:
        """
        透過 ID 讀取報告
        回傳: DiagnosticReport 物件 或 None
        """
        try:
            doc_ref = self.db.collection(self.collection).document(report_id)
            doc = doc_ref.get()

            if not doc.exists:
                print(f"⚠️ Report not found: {report_id}")
                return None

            data = doc.to_dict()
            
            # Dict 轉回 Pydantic 物件 (這一步會自動驗證資料結構)
            return DiagnosticReport(**data)
            
        except Exception as e:
            print(f"❌ Failed to get report {report_id}: {e}")
            return None

    def get_reports_by_user(self, user_id: str, limit: int = 5) -> list[DiagnosticReport]:
        """
        取得特定使用者的歷史紀錄
        """
        try:
            docs = (
                self.db.collection(self.collection)
                .where(field_path="user_id", op_string="==", value=user_id)
                .order_by("timestamp", direction="DESCENDING")
                .limit(limit)
                .stream()
            )
            return [DiagnosticReport(**doc.to_dict()) for doc in docs]
        except Exception as e:
            print(f"❌ Error fetching user history: {e}")
            return []
    
    def save_user_state(self, user_id: str, data: dict):
        """儲存使用者 Persona 與問卷暫存狀態"""
        try:
            self.db.collection("user_states").document(user_id).set(data, merge=True)
        except Exception as e:
            print(f"❌ Save user state failed: {e}")

    def get_user_state(self, user_id: str) -> dict:
        """讀取使用者狀態，若無則回傳預設值"""
        try:
            doc = self.db.collection("user_states").document(user_id).get()
            if doc.exists:
                return doc.to_dict()
            return {"persona": "doctor", "survey": None} # 預設值
        except Exception as e:
            print(f"❌ Get user state failed: {e}")
            return {"persona": "doctor", "survey": None}

# 單例模式實例化
db_service = DatabaseService()