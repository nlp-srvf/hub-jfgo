import uuid
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, jsonify, request

from task_dialogue_system import DialogueSystem


BASE_DIR = Path(__file__).parent
EXCEL_CANDIDATES = [BASE_DIR / "slot_ontology.xlsx", BASE_DIR / "slot_ontology.xls"]
EXCEL_PATH = next((p for p in EXCEL_CANDIDATES if p.exists()), None)

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# 会话数据保存在内存中，适合本地开发与演示。
SESSIONS: Dict[str, DialogueSystem] = {}


def build_system() -> DialogueSystem:
    return DialogueSystem(BASE_DIR / "scenario", excel_path=EXCEL_PATH)


def safe_get_system(session_id: Optional[str]) -> Optional[DialogueSystem]:
    if not session_id:
        return None
    return SESSIONS.get(session_id)


@app.route("/api/scenarios", methods=["GET"])
def list_scenarios():
    ds = build_system()
    return jsonify({"scenarios": ds.list_scenarios()})


@app.route("/api/session/start", methods=["POST"])
def start_session():
    payload = request.get_json(silent=True) or {}
    scenario_name = str(payload.get("scenario_name", "")).strip()
    if not scenario_name:
        return jsonify({"error": "缺少 scenario_name"}), 400

    ds = build_system()
    try:
        ds.start(scenario_name)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = ds
    welcome = f"已进入场景：{scenario_name}。请先说出你的需求。"
    return jsonify({"session_id": session_id, "welcome": welcome})


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("session_id", "")).strip()
    message = str(payload.get("message", "")).strip()
    if not session_id or not message:
        return jsonify({"error": "缺少 session_id 或 message"}), 400

    ds = safe_get_system(session_id)
    if not ds:
        return jsonify({"error": "会话不存在或已过期，请重新开始场景。"}), 404

    reply = ds.chat(message)
    finished = bool(ds.state and ds.state.is_finished)
    return jsonify({"reply": reply, "finished": finished})


@app.route("/", methods=["GET"])
def index():
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>任务型多轮对话</title>
  <style>
    :root {
      --bg: #f4f6fb;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --user: #2563eb;
      --assistant: #e5e7eb;
      --assistant-text: #111827;
      --border: #e5e7eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    .app {
      max-width: 900px;
      height: 100vh;
      margin: 0 auto;
      display: grid;
      grid-template-rows: auto 1fr auto;
      padding: 16px;
      gap: 12px;
    }
    .topbar {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .title {
      font-weight: 700;
      margin-right: 6px;
    }
    select, button, textarea {
      border: 1px solid var(--border);
      border-radius: 10px;
      font-size: 14px;
    }
    select, button { height: 36px; padding: 0 12px; }
    button {
      background: #111827;
      color: #fff;
      cursor: pointer;
    }
    button:disabled {
      opacity: .5;
      cursor: not-allowed;
    }
    .chat {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .msg {
      max-width: 82%;
      padding: 10px 12px;
      border-radius: 12px;
      white-space: pre-wrap;
      line-height: 1.45;
    }
    .msg.user {
      align-self: flex-end;
      background: var(--user);
      color: #fff;
      border-bottom-right-radius: 4px;
    }
    .msg.assistant {
      align-self: flex-start;
      background: var(--assistant);
      color: var(--assistant-text);
      border-bottom-left-radius: 4px;
    }
    .composer {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: end;
    }
    textarea {
      width: 100%;
      min-height: 64px;
      max-height: 180px;
      resize: vertical;
      padding: 10px;
      font: inherit;
      color: var(--text);
    }
    .hint {
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="topbar">
      <span class="title">任务型多轮对话</span>
      <select id="scenarioSelect"></select>
      <button id="startBtn">开始新会话</button>
      <span class="hint" id="status">未开始</span>
    </div>
    <div class="chat" id="chatBox"></div>
    <div class="composer">
      <div>
        <textarea id="inputBox" placeholder="输入消息，Enter 发送，Shift+Enter 换行"></textarea>
        <div class="hint">当前是本地内存会话，刷新页面后需重新开始会话。</div>
      </div>
      <button id="sendBtn">发送</button>
    </div>
  </div>

  <script>
    const chatBox = document.getElementById("chatBox");
    const scenarioSelect = document.getElementById("scenarioSelect");
    const startBtn = document.getElementById("startBtn");
    const sendBtn = document.getElementById("sendBtn");
    const inputBox = document.getElementById("inputBox");
    const statusEl = document.getElementById("status");

    let sessionId = "";
    let finished = false;

    function addMessage(role, text) {
      const div = document.createElement("div");
      div.className = "msg " + role;
      div.textContent = text;
      chatBox.appendChild(div);
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function loadScenarios() {
      try {
        const resp = await fetch("/api/scenarios");
        console.log("场景列表响应状态:", resp.status);
        const data = await resp.json();
        console.log("场景列表数据:", data);
        scenarioSelect.innerHTML = "";
        (data.scenarios || []).forEach((name) => {
          const op = document.createElement("option");
          op.value = name;
          op.textContent = name;
          scenarioSelect.appendChild(op);
        });
        console.log("场景选择器值:", scenarioSelect.value);
        if (!scenarioSelect.value) {
          statusEl.textContent = "未发现可用场景 JSON";
        } else {
          statusEl.textContent = "场景加载成功";
        }
      } catch (error) {
        console.error("加载场景失败:", error);
        statusEl.textContent = "加载场景失败: " + error.message;
      }
    }

    async function startSession() {
      const scenarioName = scenarioSelect.value;
      if (!scenarioName) return;

      const resp = await fetch("/api/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario_name: scenarioName }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        addMessage("assistant", "启动失败：" + (data.error || "未知错误"));
        return;
      }
      sessionId = data.session_id;
      finished = false;
      chatBox.innerHTML = "";
      statusEl.textContent = "会话进行中（场景：" + scenarioName + "）";
      addMessage("assistant", data.welcome || "会话已开始。");
      inputBox.focus();
    }

    async function sendMessage() {
      const message = inputBox.value.trim();
      if (!message) return;
      if (!sessionId) {
        addMessage("assistant", "请先点击“开始新会话”。");
        return;
      }
      if (finished) {
        addMessage("assistant", "流程已结束，请开始新会话。");
        return;
      }

      addMessage("user", message);
      inputBox.value = "";
      sendBtn.disabled = true;
      try {
        const resp = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, message }),
        });
        const data = await resp.json();
        if (!resp.ok) {
          addMessage("assistant", "请求失败：" + (data.error || "未知错误"));
          return;
        }
        addMessage("assistant", data.reply || "");
        finished = !!data.finished;
        if (finished) {
          statusEl.textContent = "会话已结束";
        }
      } finally {
        sendBtn.disabled = false;
        inputBox.focus();
      }
    }

    inputBox.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
    sendBtn.addEventListener("click", sendMessage);
    startBtn.addEventListener("click", startSession);

    loadScenarios().then(() => {
      if (scenarioSelect.value) {
        startSession();
      }
    });
  </script>
</body>
</html>"""


if __name__ == "__main__":
    try:
        print("启动Web服务...")
        print(f"场景目录: {BASE_DIR / 'scenario'}")
        print(f"场景目录是否存在: {(BASE_DIR / 'scenario').exists()}")
        # 测试场景加载
        test_ds = build_system()
        print(f"成功加载场景: {test_ds.list_scenarios()}")
        print("启动Flask应用...")
        app.run(host="127.0.0.1", port=8000, debug=False)
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
        raise
