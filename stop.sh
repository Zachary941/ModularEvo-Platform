#!/bin/bash
# TransModular Demo 一键停止脚本
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BACKEND_PID_FILE="$SCRIPT_DIR/.backend.pid"
FRONTEND_PID_FILE="$SCRIPT_DIR/.frontend.pid"

echo "=== TransModular Demo 停止 ==="

# 停止后端
if [ -f "$BACKEND_PID_FILE" ]; then
    PID=$(cat "$BACKEND_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null
        echo "[1/2] 后端已停止 (PID: $PID)"
    else
        echo "[1/2] 后端进程已不存在 (PID: $PID)"
    fi
    rm -f "$BACKEND_PID_FILE"
else
    echo "[1/2] 未找到后端 PID 文件，尝试 pkill ..."
    pkill -f "uvicorn main:app.*8000" 2>/dev/null && echo "      已停止" || echo "      无运行进程"
fi

# 停止前端
if [ -f "$FRONTEND_PID_FILE" ]; then
    PID=$(cat "$FRONTEND_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null
        echo "[2/2] 前端已停止 (PID: $PID)"
    else
        echo "[2/2] 前端进程已不存在 (PID: $PID)"
    fi
    rm -f "$FRONTEND_PID_FILE"
else
    echo "[2/2] 未找到前端 PID 文件，尝试 pkill ..."
    pkill -f "vite.*3000" 2>/dev/null && echo "      已停止" || echo "      无运行进程"
fi

echo ""
echo "=== 已停止所有服务 ==="
