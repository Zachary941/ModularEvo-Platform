#!/bin/bash
# TransModular Demo 一键启动脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="/home/longwr/.local/bin:$PATH"
export DEMO_PROJECT_ROOT="$SCRIPT_DIR"
export DEMO_ALGO_ROOT="$(dirname "$SCRIPT_DIR")"

# 查找可用的 node 二进制
NODE_BIN=""
if command -v node >/dev/null 2>&1; then
    NODE_BIN="$(command -v node)"
else
    # 使用 VS Code Server 自带的 node
    NODE_BIN="$(find /home/longwr/.vscode-server/cli/servers -name 'node' -type f 2>/dev/null | head -1)"
fi
if [ -z "$NODE_BIN" ]; then
    echo "[ERROR] 未找到 node 二进制，请安装 Node.js"
    exit 1
fi
NODE_DIR="$(dirname "$NODE_BIN")"
export PATH="$NODE_DIR:$PATH"

BACKEND_PID_FILE="$SCRIPT_DIR/.backend.pid"
FRONTEND_PID_FILE="$SCRIPT_DIR/.frontend.pid"

# 检查端口是否被占用
check_port() {
    if lsof -i :"$1" >/dev/null 2>&1 || (command -v ss >/dev/null 2>&1 && ss -tlnp 2>/dev/null | grep -q ":$1 "); then
        echo "[ERROR] 端口 $1 已被占用，请先运行 stop.sh 或手动释放端口"
        return 1
    fi
    return 0
}

echo "=== TransModular Demo 启动 ==="

# 检查端口
check_port 8000 || exit 1
check_port 3000 || exit 1

# 启动后端
echo "[1/2] 启动后端 (FastAPI on :8000) ..."
cd "$SCRIPT_DIR/backend"
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > "$SCRIPT_DIR/.backend.log" 2>&1 &
echo $! > "$BACKEND_PID_FILE"
echo "      PID: $(cat "$BACKEND_PID_FILE")"

# 等待后端就绪
for i in $(seq 1 10); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "      后端就绪 ✓"
        break
    fi
    sleep 1
done

# 启动前端
echo "[2/2] 启动前端 (Vite on :3000) ..."
cd "$SCRIPT_DIR/frontend"
VITE_BIN="$SCRIPT_DIR/frontend/node_modules/.bin/vite"
nohup "$NODE_BIN" "$VITE_BIN" --port 3000 --host 0.0.0.0 > "$SCRIPT_DIR/.frontend.log" 2>&1 &
echo $! > "$FRONTEND_PID_FILE"
echo "      PID: $(cat "$FRONTEND_PID_FILE")"
sleep 2

echo ""
echo "=== 启动完成 ==="
echo "  后端: http://localhost:8000  (API docs: http://localhost:8000/docs)"
echo "  前端: http://localhost:3000"
echo "  日志: $SCRIPT_DIR/.backend.log / .frontend.log"
echo "  停止: bash $SCRIPT_DIR/stop.sh"
