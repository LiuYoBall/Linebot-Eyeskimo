# 1. 使用官方 Python 3.11 輕量版
FROM python:3.11-slim

# === 安裝系統級依賴 (防呆) ===
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. 從官方 image 複製 uv執行檔
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 3. 設定工作目錄
WORKDIR /app
# === 設定環境變數，將虛擬環境加入 PATH ===
ENV PATH="/app/.venv/bin:$PATH"

# 4. 複製依賴描述檔
COPY pyproject.toml uv.lock ./

# 5. uv sync 會自動在 /app/.venv 建立虛擬環境
RUN uv sync --frozen

# 6. 複製您的程式碼與靜態資源
COPY . .

RUN echo "=== Checking weights directory ===" && \
    ls -lh weights/ || echo "weights directory NOT FOUND" && \
    echo "=================================="

# 7. 啟動指令 (使用 fastAPI)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]