"""YHJ Agent Streamlit 前端。

核心功能：
1. 动态执行流程图（Graphviz）
2. Clarification interrupt/resume 交互
3. 最终报告展示
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any

import streamlit as st

# ── 配置 ──────────────────────────────────────────────────────────

API_BASE = "http://127.0.0.1:8001"

# 节点标签
NODE_LABELS = {
    "begin": "Begin",
    "input_normalizer": "Input",
    "intent_router": "Intent",
    "profile_memory_reader": "Profile",
    "planner_executor": "Plan",
    "symbolic_calculator": "Bazi",
    "domain_rag": "RAG",
    "specialist_subgraph": "Specialist",
    "conflict_debate": "Debate",
    "synthesis": "Synthesis",
    "critic_evaluator": "Critic",
    "report_generator": "Report",
    "memory_writer": "Memory",
    "explanation_node": "Explain",
    "report_compressor": "Compress",
    "safe_completion": "Safe",
    "clarification_node": "Clarify",
    "done": "Done",
}

# 10 种执行路径的完整节点序列和边
PATH_DATA: dict[str, dict] = {
    "new_task": {
        "nodes": ["begin", "input_normalizer", "intent_router", "profile_memory_reader",
                  "planner_executor", "symbolic_calculator", "domain_rag",
                  "specialist_subgraph", "conflict_debate", "synthesis",
                  "critic_evaluator", "report_generator", "memory_writer", "done"],
        "edges": [
            ("begin", "input_normalizer"),
            ("input_normalizer", "intent_router"),
            ("intent_router", "profile_memory_reader"),
            ("profile_memory_reader", "planner_executor"),
            ("planner_executor", "symbolic_calculator"),
            ("planner_executor", "domain_rag"),
            ("symbolic_calculator", "specialist_subgraph"),
            ("domain_rag", "specialist_subgraph"),
            ("specialist_subgraph", "conflict_debate"),
            ("conflict_debate", "synthesis"),
            ("synthesis", "critic_evaluator"),
            ("critic_evaluator", "report_generator"),
            ("report_generator", "memory_writer"),
            ("memory_writer", "done"),
        ],
    },
    "follow_up_question": {
        "nodes": ["begin", "input_normalizer", "intent_router", "planner_executor",
                  "explanation_node", "memory_writer", "done"],
        "edges": [
            ("begin", "input_normalizer"),
            ("input_normalizer", "intent_router"),
            ("intent_router", "planner_executor"),
            ("planner_executor", "explanation_node"),
            ("explanation_node", "memory_writer"),
            ("memory_writer", "done"),
        ],
    },
    "correction": {
        "nodes": ["begin", "input_normalizer", "intent_router", "planner_executor",
                  "symbolic_calculator", "domain_rag", "specialist_subgraph",
                  "conflict_debate", "synthesis", "critic_evaluator",
                  "report_generator", "memory_writer", "done"],
        "edges": [
            ("begin", "input_normalizer"),
            ("input_normalizer", "intent_router"),
            ("intent_router", "planner_executor"),
            ("planner_executor", "symbolic_calculator"),
            ("planner_executor", "domain_rag"),
            ("symbolic_calculator", "specialist_subgraph"),
            ("domain_rag", "specialist_subgraph"),
            ("specialist_subgraph", "conflict_debate"),
            ("conflict_debate", "synthesis"),
            ("synthesis", "critic_evaluator"),
            ("critic_evaluator", "report_generator"),
            ("report_generator", "memory_writer"),
            ("memory_writer", "done"),
        ],
    },
    "topic_switch": {
        "nodes": ["begin", "input_normalizer", "intent_router", "planner_executor",
                  "symbolic_calculator", "domain_rag", "specialist_subgraph",
                  "conflict_debate", "synthesis", "critic_evaluator",
                  "report_generator", "memory_writer", "done"],
        "edges": [
            ("begin", "input_normalizer"),
            ("input_normalizer", "intent_router"),
            ("intent_router", "planner_executor"),
            ("planner_executor", "symbolic_calculator"),
            ("planner_executor", "domain_rag"),
            ("symbolic_calculator", "specialist_subgraph"),
            ("domain_rag", "specialist_subgraph"),
            ("specialist_subgraph", "conflict_debate"),
            ("conflict_debate", "synthesis"),
            ("synthesis", "critic_evaluator"),
            ("critic_evaluator", "report_generator"),
            ("report_generator", "memory_writer"),
            ("memory_writer", "done"),
        ],
    },
    "format_refinement": {
        "nodes": ["begin", "input_normalizer", "intent_router", "planner_executor",
                  "report_compressor", "memory_writer", "done"],
        "edges": [
            ("begin", "input_normalizer"),
            ("input_normalizer", "intent_router"),
            ("intent_router", "planner_executor"),
            ("planner_executor", "report_compressor"),
            ("report_compressor", "memory_writer"),
            ("memory_writer", "done"),
        ],
    },
    "safety_intervention": {
        "nodes": ["begin", "input_normalizer", "intent_router", "safe_completion",
                  "memory_writer", "done"],
        "edges": [
            ("begin", "input_normalizer"),
            ("input_normalizer", "intent_router"),
            ("intent_router", "safe_completion"),
            ("safe_completion", "memory_writer"),
            ("memory_writer", "done"),
        ],
    },
    "clarification": {
        "nodes": ["begin", "input_normalizer", "intent_router", "clarification_node",
                  "intent_router", "profile_memory_reader", "planner_executor",
                  "symbolic_calculator", "domain_rag", "specialist_subgraph",
                  "conflict_debate", "synthesis", "critic_evaluator",
                  "report_generator", "memory_writer", "done"],
        "edges": [
            ("begin", "input_normalizer"),
            ("input_normalizer", "intent_router"),
            ("intent_router", "clarification_node"),
            ("clarification_node", "intent_router"),
            ("intent_router", "profile_memory_reader"),
            ("profile_memory_reader", "planner_executor"),
            ("planner_executor", "symbolic_calculator"),
            ("planner_executor", "domain_rag"),
            ("symbolic_calculator", "specialist_subgraph"),
            ("domain_rag", "specialist_subgraph"),
            ("specialist_subgraph", "conflict_debate"),
            ("conflict_debate", "synthesis"),
            ("synthesis", "critic_evaluator"),
            ("critic_evaluator", "report_generator"),
            ("report_generator", "memory_writer"),
            ("memory_writer", "done"),
        ],
    },
    "explanation": {
        "nodes": ["begin", "input_normalizer", "intent_router", "planner_executor",
                  "explanation_node", "memory_writer", "done"],
        "edges": [
            ("begin", "input_normalizer"),
            ("input_normalizer", "intent_router"),
            ("intent_router", "planner_executor"),
            ("planner_executor", "explanation_node"),
            ("explanation_node", "memory_writer"),
            ("memory_writer", "done"),
        ],
    },
    "reuse_cached": {
        "nodes": ["begin", "input_normalizer", "intent_router",
                  "planner_executor", "report_generator", "memory_writer", "done"],
        "edges": [
            ("begin", "input_normalizer"),
            ("input_normalizer", "intent_router"),
            ("intent_router", "planner_executor"),
            ("planner_executor", "report_generator"),
            ("report_generator", "memory_writer"),
            ("memory_writer", "done"),
        ],
    },
    "conflict_debate": {
        "nodes": ["begin", "input_normalizer", "intent_router", "profile_memory_reader",
                  "planner_executor", "symbolic_calculator", "domain_rag",
                  "specialist_subgraph", "conflict_debate", "synthesis",
                  "critic_evaluator", "conflict_debate", "synthesis",
                  "critic_evaluator", "report_generator", "memory_writer", "done"],
        "edges": [
            ("begin", "input_normalizer"),
            ("input_normalizer", "intent_router"),
            ("intent_router", "profile_memory_reader"),
            ("profile_memory_reader", "planner_executor"),
            ("planner_executor", "symbolic_calculator"),
            ("planner_executor", "domain_rag"),
            ("symbolic_calculator", "specialist_subgraph"),
            ("domain_rag", "specialist_subgraph"),
            ("specialist_subgraph", "conflict_debate"),
            ("conflict_debate", "synthesis"),
            ("synthesis", "critic_evaluator"),
            ("critic_evaluator", "conflict_debate"),
            ("conflict_debate", "synthesis"),
            ("synthesis", "critic_evaluator"),
            ("critic_evaluator", "report_generator"),
            ("report_generator", "memory_writer"),
            ("memory_writer", "done"),
        ],
    },
}

# 路径中文名
PATH_NAMES = {
    "new_task": "新任务",
    "follow_up_question": "追问",
    "correction": "纠错",
    "topic_switch": "主题切换",
    "format_refinement": "格式修改",
    "safety_intervention": "安全降级",
    "clarification": "澄清补全",
    "explanation": "解释说明",
    "reuse_cached": "缓存命中",
    "conflict_debate": "冲突辩论",
}

# turn_type 映射
TURN_TYPE_TO_PATH = {
    "new_task": "new_task",
    "follow_up_question": "follow_up_question",
    "correction": "correction",
    "topic_switch": "topic_switch",
    "format_refinement": "format_refinement",
    "safety_intervention": "safety_intervention",
    "missing_field": "clarification",
    "clarification": "clarification",
    "clarification_resolved": "clarification",
}


# ── API 调用 ──────────────────────────────────────────────────────

def api_post(path: str, data: dict) -> dict:
    """调用 POST API。"""
    url = f"{API_BASE}{path}"
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=200) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"error": str(e)}




def api_post_async(path: str, data: dict) -> dict:
    """发送 POST 请求，立即返回（用于创建异步任务）。"""
    url = f"{API_BASE}{path}"
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"error": str(e)}


def api_get(path: str) -> dict:
    """发送 GET 请求。"""
    url = f"{API_BASE}{path}"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"error": str(e)}


def api_poll_task(task_id: str, interval: float = 3.0, max_wait: float = 600.0) -> dict:
    """轮询异步任务状态，返回最终结果。"""
    import time as _time
    start = _time.time()
    while _time.time() - start < max_wait:
        resp = api_get(f"/api/tasks/{task_id}")
        if resp.get("error"):
            return resp
        status = resp.get("status", "")
        if status == "completed":
            return resp.get("result", {})
        elif status == "failed":
            return resp.get("result", {"error": "任务执行失败"})
        _time.sleep(interval)
    return {"error": f"任务超时（等待 {max_wait:.0f} 秒）"}

# ── 路径推断 ──────────────────────────────────────────────────────

def infer_path_key(data: dict, clarification_resolved: bool = False) -> str:
    """推断执行路径的 key。"""
    if data.get("interrupted"):
        return "clarification"

    if clarification_resolved:
        return "clarification"

    # execution_mode 优先（explanation、reuse_cached 是 execution_mode，不是 turn_type）
    execution_mode = data.get("execution_mode", "")
    if execution_mode in ("explanation", "reuse_cached"):
        return execution_mode

    turn_type = data.get("turn_type", "")
    return TURN_TYPE_TO_PATH.get(turn_type, "new_task")


def get_path_info(path_key: str) -> dict:
    """获取路径信息（节点和边）。"""
    return PATH_DATA.get(path_key, PATH_DATA["new_task"])


# ── 流程图渲染 ────────────────────────────────────────────────────

def build_graphviz_dot(path_key: str) -> str:
    """生成 Graphviz DOT 语言的流程图。

    对于有重复节点的路径（如 clarification、conflict_debate），
    使用 _1、_2 后缀作为唯一节点 ID，但保持相同的显示标签。

    边的映射逻辑：
    1. 为节点序列中的每个位置生成唯一 ID
    2. 对于每条边，根据 src/dst 在节点序列中的出现顺序确定唯一 ID
    """
    path_info = get_path_info(path_key)
    raw_nodes = path_info["nodes"]
    raw_edges = path_info["edges"]

    # 为每个位置的节点生成唯一 ID
    node_occurrence: dict[str, int] = {}
    node_ids: list[str] = []

    for node in raw_nodes:
        count = node_occurrence.get(node, 0)
        if count == 0:
            unique_id = node
        else:
            unique_id = f"{node}_{count + 1}"
        node_occurrence[node] = count + 1
        node_ids.append(unique_id)

    # 构建节点位置索引：记录每个唯一 ID 在序列中的位置
    node_position: dict[str, int] = {}
    for i, uid in enumerate(node_ids):
        node_position[uid] = i

    # 为每条边确定正确的唯一 ID
    # 策略：对于每条边 (src, dst)，在节点序列中找到 src 和 dst 的位置
    # 使用位置来确定使用哪个唯一 ID
    edge_pairs: list[tuple[str, str]] = []
    src_used: dict[str, int] = {}  # 记录每个 src 节点已使用的次数
    dst_used: dict[str, int] = {}  # 记录每个 dst 节点已使用的次数

    for src, dst in raw_edges:
        # 获取 src 的第 N 次出现的唯一 ID
        src_count = src_used.get(src, 0)
        src_occ = node_occurrence.get(src, 0)

        if src_occ == 1:
            # 只出现一次，直接使用原始 ID
            src_id = src
        else:
            # 出现多次，使用第 N 次出现的唯一 ID
            if src_count == 0:
                src_id = src
            else:
                src_id = f"{src}_{src_count + 1}"
        src_used[src] = src_count + 1

        # 获取 dst 的第 N 次出现的唯一 ID
        dst_count = dst_used.get(dst, 0)
        dst_occ = node_occurrence.get(dst, 0)

        if dst_occ == 1:
            # 只出现一次，直接使用原始 ID
            dst_id = dst
        else:
            # 出现多次，使用第 N 次出现的唯一 ID
            if dst_count == 0:
                dst_id = dst
            else:
                dst_id = f"{dst}_{dst_count + 1}"
        dst_used[dst] = dst_count + 1

        edge_pairs.append((src_id, dst_id))

    # 去重边（保持顺序）
    seen_edges = set()
    unique_edges = []
    for src_id, dst_id in edge_pairs:
        edge_key = f"{src_id}->{dst_id}"
        if edge_key not in seen_edges:
            unique_edges.append((src_id, dst_id))
            seen_edges.add(edge_key)

    lines = [
        'digraph {',
        '  rankdir=LR;',
        '  node [shape=box, style="rounded,filled", fontname="Microsoft YaHei", fontsize=11, width=1.0];',
        '  edge [color="#28a745", penwidth=2];',
        '  bgcolor="transparent";',
        '',
    ]

    # 渲染节点（使用唯一 ID，但标签使用原始名称）
    seen_nodes = set()
    for unique_id in node_ids:
        if unique_id not in seen_nodes:
            # 从唯一 ID 提取原始节点名
            base_name = unique_id.rsplit("_", 1)[0] if "_" in unique_id else unique_id
            label = NODE_LABELS.get(base_name, NODE_LABELS.get(unique_id, base_name))
            color = "#d4edda" if base_name not in ("begin", "done") else "#fff3cd"
            border = "#28a745" if base_name not in ("begin", "done") else "#ffc107"
            lines.append(f'  {unique_id} [label="{label}", fillcolor="{color}", color="{border}"];')
            seen_nodes.add(unique_id)

    lines.append('')

    # 渲染边
    for src_id, dst_id in unique_edges:
        lines.append(f'  {src_id} -> {dst_id};')

    lines.append('}')
    return '\n'.join(lines)


# ── 页面配置 ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="State-Aware Multi-Turn Agent System",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 状态感知多轮 Agent 执行系统")
st.caption("支持八字命理、RAG 检索、专家辩论、Critic 修订等能力")


# ── Session State 初始化 ─────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-demo"
if "clarification_pending" not in st.session_state:
    st.session_state.clarification_pending = False
if "clarification_question" not in st.session_state:
    st.session_state.clarification_question = ""
if "clarification_resolved" not in st.session_state:
    st.session_state.clarification_resolved = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ── 侧边栏 ───────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ 设置")
    session_id = st.text_input("Session ID", value=st.session_state.session_id)
    if session_id != st.session_state.session_id:
        st.session_state.session_id = session_id
        st.session_state.messages = []
        st.session_state.clarification_pending = False
        st.session_state.clarification_resolved = False
        st.session_state.last_result = None
        st.rerun()

    if st.button("🆕 新建会话"):
        import uuid as _uuid; st.session_state.session_id = f"session-{_uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.session_state.clarification_pending = False
        st.session_state.clarification_resolved = False
        st.session_state.last_result = None
        st.rerun()

    st.divider()
    st.header("📖 10 种执行路径")
    st.markdown("""
    | 路径 | 触发条件 | 节点数 |
    |------|---------|--------|
    | 🆕 新任务 | 完整咨询请求 | 14 |
    | ❓ 追问 | 基于上一轮追问 | 7 |
    | ✏️ 纠错 | 指出之前错误 | 13 |
    | 🔄 主题切换 | 切换咨询主题 | 13 |
    | 📝 格式修改 | 修改报告格式 | 7 |
    | 🚨 安全降级 | 高风险输入 | 6 |
    | 💬 澄清补全 | 缺失关键字段 | 16 |
    | 📖 解释说明 | 请求解释概念 | 7 |
    | ⚡ 缓存命中 | 出生信息未变 | 8 |
    | ⚔️ 冲突辩论 | 专家意见冲突 | 14 |
    """)


# ── 主区域：左右分栏 ─────────────────────────────────────────────

col_chat, col_flow = st.columns([1, 1])

# ── 左栏：对话交互 ────────────────────────────────────────────────

with col_chat:
    st.subheader("💬 对话")

    # 对话历史（折叠）
    with st.expander("📜 对话历史", expanded=False):
        if not st.session_state.messages:
            st.caption("暂无对话记录")
        else:
            for msg in st.session_state.messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    st.markdown(f"**🧑 用户：** {content}")
                else:
                    st.markdown(f"**🤖 Agent：** {content}")
                st.divider()

    # Clarification 交互
    if st.session_state.clarification_pending:
        st.warning(f"**系统需要补充信息：**\n\n{st.session_state.clarification_question}")
        clarification_answer = st.text_input("请输入补充信息：", key="clarification_input")
        if st.button("📤 提交补充信息", type="primary"):
            if clarification_answer.strip():
                st.session_state.messages.append({"role": "user", "content": f"[补充] {clarification_answer}"})
                with st.spinner("正在恢复执行..."):
                    task_resp = api_post_async("/api/tasks", {
                        "type": "resume",
                        "session_id": st.session_state.session_id,
                        "answer": clarification_answer.strip(),
                    })
                    if task_resp.get("error"):
                        result = task_resp
                    else:
                        result = api_poll_task(task_resp["task_id"])
                st.session_state.clarification_pending = False
                st.session_state.clarification_question = ""
                st.session_state.last_result = result

                if result.get("error"):
                    st.error(f"恢复失败：{result['error']}")
                elif result.get("interrupted"):
                    # 再次 interrupt（连续澄清）
                    st.session_state.clarification_pending = True
                    st.session_state.clarification_question = result.get("question", "请继续补充")
                    st.rerun()
                else:
                    # 澄清恢复成功
                    st.session_state.clarification_resolved = True
                    report = result.get("final_report", "")
                    st.session_state.messages.append({"role": "assistant", "content": report})
                    st.rerun()
    else:
        # 普通输入
        user_input = st.text_area(
            "输入您的问题：",
            value="",
            height=120,
            placeholder="例如：我是2000年3月4日巳时男重庆出生，事业怎么判断？",
        )
        if st.button("🚀 发送", type="primary"):
            if user_input.strip():
                st.session_state.messages.append({"role": "user", "content": user_input.strip()})
                # 新的一轮，重置 clarification_resolved
                st.session_state.clarification_resolved = False
                with st.spinner("正在分析..."):
                    task_resp = api_post_async("/api/tasks", {
                        "type": "turn",
                        "session_id": st.session_state.session_id,
                        "message": user_input.strip(),
                    })
                    if task_resp.get("error"):
                        result = task_resp
                    else:
                        result = api_poll_task(task_resp["task_id"])
                st.session_state.last_result = result

                if result.get("error"):
                    st.error(f"请求失败：{result['error']}")
                elif result.get("interrupted"):
                    st.session_state.clarification_pending = True
                    st.session_state.clarification_question = result.get("question", "请补充信息")
                    st.rerun()
                else:
                    report = result.get("final_report", "")
                    st.session_state.messages.append({"role": "assistant", "content": report})
                    st.rerun()


# ── 右栏：执行流程 + 最终报告 ─────────────────────────────────────

with col_flow:
    st.subheader("📊 执行流程")

    result = st.session_state.last_result

    if result is None:
        st.info("输入问题后查看执行流程")
    elif result.get("error"):
        st.error(f"执行失败：{result['error']}")
    elif result.get("interrupted"):
        st.warning("⏸️ 执行暂停 — 等待用户补充信息")
        dot = build_graphviz_dot("clarification")
        st.graphviz_chart(dot, width='stretch')
    else:
        # 正常完成
        path_key = infer_path_key(result, st.session_state.clarification_resolved)
        path_info = get_path_info(path_key)
        path_name = PATH_NAMES.get(path_key, path_key)

        # 统计 unique 节点数和 unique 边数
        unique_nodes = len(set(path_info["nodes"]))
        unique_edges = len(set(path_info["edges"]))

        st.success(f"✅ 路径：**{path_name}**（{unique_nodes} 个节点, {unique_edges} 条边）")

        # 流程图
        dot = build_graphviz_dot(path_key)
        st.graphviz_chart(dot, width='stretch')

        # ── 执行元数据 ─────────────────────────────────────────
        st.divider()
        st.subheader("📋 执行元数据")

        # 数据准备
        req = result.get("consultation_request", {}) or {}
        evaluation = result.get("evaluation", {}) or {}
        symbolic_result = result.get("symbolic_result", {}) or {}
        sym_result = symbolic_result if isinstance(symbolic_result, dict) else {}
        debate_output = result.get("debate_output", {}) or {}
        synthesis = result.get("synthesis", {}) or {}
        execution_plan = result.get("execution_plan", {}) or {}
        mode = execution_plan.get("execution_mode", "")

        turn_type = result.get("turn_type", "-")
        risk_level = result.get("risk_level", 0)
        overall_score = evaluation.get("overall_score", 0)

        # 显示名映射
        TURN_TYPE_DISPLAY = {
            "new_task": "新任务", "follow_up_question": "追问", "correction": "纠错",
            "topic_switch": "主题切换", "format_refinement": "格式修改",
            "safety_intervention": "安全降级", "missing_field": "澄清补全",
            "clarification": "澄清补全", "clarification_resolved": "澄清补全",
            "explanation": "解释说明", "reuse_cached": "缓存命中",
        }
        MODE_DISPLAY = {
            "full_execution": "完整执行", "explanation_only": "仅解释",
            "format_only": "仅格式", "style_only": "仅风格",
            "safe_completion": "安全降级", "reuse_cached": "缓存复用",
        }


        # 轮次类型与路径对齐：当路径被 infer_path_key 提升为澄清补全时，
        # 元数据也应显示澄清补全而非原始 turn_type
        if path_key == "clarification" and turn_type not in ("missing_field", "clarification", "clarification_resolved"):
            turn_type = "clarification"

        # L0: 4 个 metric 卡片
        row1_c1, row1_c2 = st.columns(2)
        with row1_c1:
            st.metric("轮次类型", TURN_TYPE_DISPLAY.get(turn_type, turn_type))
        with row1_c2:
            st.metric("执行模式", MODE_DISPLAY.get(mode, mode or "-"))
        row2_c1, row2_c2 = st.columns(2)
        with row2_c1:
            if risk_level >= 7:
                st.error(f"风险等级: {risk_level} ⚠️")
            elif risk_level >= 4:
                st.warning(f"风险等级: {risk_level}")
            else:
                st.metric("风险等级", risk_level)
        with row2_c2:
            if overall_score:
                st.metric("Critic 评分", f"{overall_score:.1f}/5")
            else:
                st.metric("Critic 评分", "-")


        # L1: 可折叠详情
        has_evaluation = bool(evaluation.get("overall_score"))
        has_symbolic = bool(sym_result.get("day_master"))
        has_debate = bool(debate_output.get("debate_occurred"))
        has_synthesis = bool(synthesis.get("confidence"))
        has_detail = has_evaluation or has_symbolic or has_debate or has_synthesis

        if has_detail:
            with st.expander("📊 详细评分与执行详情", expanded=False):
                tab_eval, tab_bazi, tab_reasoning = st.tabs(["评分详情", "八字命理", "推理过程"])

                # Tab 1: 评分详情
                with tab_eval:
                    if has_evaluation:
                        for dim, label in [
                            ("evidence_score", "证据充分性"),
                            ("safety_score", "安全性"),
                            ("practicality_score", "实用性"),
                            ("balance_score", "平衡性"),
                            ("actionability_score", "可操作性"),
                        ]:
                            score = evaluation.get(dim, 0)
                            st.progress(score / 5.0, text=f"{label}: {score:.1f}/5")

                        need_revision = evaluation.get("need_revision", False)
                        patch_target = evaluation.get("patch_target", "none")
                        if need_revision:
                            st.warning(f"需要修订 → 目标: {patch_target}")
                        else:
                            st.success("无需修订")
                    else:
                        st.caption("无评分数据")

                # Tab 2: 八字命理
                with tab_bazi:
                    if has_symbolic:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**日主：** {sym_result['day_master']}（{sym_result.get('day_master_strength', '-')}）")
                            favorable = sym_result.get("favorable_elements", [])
                            st.markdown(f"**喜用神：** {', '.join(favorable) if favorable else '-'}")
                            unfavorable = sym_result.get("unfavorable_elements", [])
                            st.markdown(f"**忌神：** {', '.join(unfavorable) if unfavorable else '-'}")
                        with col2:
                            current_cycle = sym_result.get("current_cycle", {})
                            if current_cycle:
                                stem = current_cycle.get("heavenly_stem", "")
                                branch = current_cycle.get("earthly_branch", "")
                                st.markdown(f"**当前大运：** {stem}{branch}")
                            current_year = sym_result.get("current_year", {})
                            if current_year:
                                stem = current_year.get("heavenly_stem", "")
                                branch = current_year.get("earthly_branch", "")
                                st.markdown(f"**流年：** {stem}{branch}")

                        ten_gods = sym_result.get("ten_gods", {})
                        if ten_gods:
                            st.markdown("**十神：**")
                            cols = st.columns(4)
                            for i, (pillar, god) in enumerate(ten_gods.items()):
                                with cols[i % 4]:
                                    st.caption(f"{pillar}: {god}")
                    else:
                        st.caption("无八字数据")

                # Tab 3: 推理过程
                with tab_reasoning:
                    if has_debate:
                        st.info(f"辩论发生，共 {debate_output.get('rounds_taken', 0)} 轮")
                        resolution = debate_output.get("resolution", {})
                        if resolution:
                            st.markdown(f"**共识：** {resolution.get('resolution_summary', '-')}")
                    else:
                        st.caption("未触发辩论")

                    if has_synthesis:
                        col1, col2 = st.columns(2)
                        with col1:
                            confidence = synthesis.get("confidence", 0)
                            st.metric("综合置信度", f"{confidence:.2f}")
                        with col2:
                            consistency = synthesis.get("consistency_score", 0)
                            st.metric("一致性分数", f"{consistency:.2f}")

        # ── 检索证据 ─────────────────────────────────────────
        evidence = result.get("domain_rag_result")
        if isinstance(evidence, dict):
            evidence = evidence.get("evidence", []) or evidence.get("results", [])
        elif not isinstance(evidence, list):
            evidence = []

        has_evidence = bool(evidence) and path_key not in (
            "follow_up_question", "format_refinement", "safety_intervention", "explanation"
        )

        if has_evidence:
            with st.expander(f"🔍 检索证据（{len(evidence)} 条）", expanded=False):
                for i, ev in enumerate(evidence, 1):
                    if not isinstance(ev, dict):
                        continue

                    col_score, col_content = st.columns([1, 4])

                    with col_score:
                        score = ev.get("score", 0)
                        if score > 0.8:
                            color = "🟢"
                        elif score > 0.5:
                            color = "🟡"
                        else:
                            color = "🔴"
                        st.metric(f"{color} 相关度", f"{score:.2f}")

                    with col_content:
                        title = ev.get("title", ev.get("doc_id", f"证据 {i}"))
                        st.markdown(f"**{title}**")
                        source = ev.get("source_name", "")
                        if source:
                            st.caption(source)
                        text = ev.get("text", "")
                        if text:
                            st.markdown(f"> {text[:300]}")

                    if i < len(evidence):
                        st.divider()

        # ── 最终报告 ─────────────────────────────────────
        st.divider()
        st.subheader("📝 最终报告")

        report = result.get("final_report", "")
        detail = result.get("final_report_detail", {}) or {}
        report_text = detail.get("report_text", "") or report

        # === 1. Summary Card ===
        summary = detail.get("summary", "")
        if summary:
            st.info(f"💡 **摘要**  {summary}")

        # === 2. Confidence Badge ===
        conf = detail.get("confidence", "")
        if conf:
            badge_map = {"high": "🟢 高置信", "medium": "🟡 中置信", "low": "🔴 低置信"}
            badge = badge_map.get(str(conf).lower(), f"⚪ {conf}")
            st.caption(f"置信度：{badge}")

        if report_text:
            st.markdown("#### 完整报告")
            st.markdown(report_text)

        # === 3. Sections (tabbed) ===
        sections = detail.get("sections", [])
        if sections and isinstance(sections, list) and len(sections) > 0:
            if len(sections) <= 6:
                tab_titles = [s.get("title", f"段落{i+1}") for i, s in enumerate(sections)]
                tabs = st.tabs(tab_titles)
                for tab, section in zip(tabs, sections):
                    with tab:
                        st.markdown(section.get("text", ""))
            else:
                for section in sections:
                    title = section.get("title", "")
                    text_content = section.get("text", "")
                    if title:
                        st.markdown(f"#### {title}")
                    if text_content:
                        st.markdown(text_content)
                    st.divider()
        elif report_text:
            pass
        else:
            st.caption("无报告内容")

        # === 4. Action Plan ===
        action_plan = detail.get("action_plan", [])
        if action_plan and isinstance(action_plan, list) and len(action_plan) > 0:
            st.markdown("#### 🎯 行动计划")
            for i, item in enumerate(action_plan, 1):
                title = item.get("title", f"行动 {i}")
                text_content = item.get("text", "")
                col_num, col_body = st.columns([1, 6])
                with col_num:
                    st.markdown(f"**{i}.**")
                with col_body:
                    st.markdown(f"**{title}**")
                    if text_content:
                        st.caption(text_content)

        # === 5. Visual Blocks ===
        visual_blocks = detail.get("visual_blocks", [])
        if visual_blocks and isinstance(visual_blocks, list) and len(visual_blocks) > 0:
            for block in visual_blocks:
                block_type = block.get("type", "text")
                block_title = block.get("title", "")
                if block_title:
                    st.markdown(f"#### 📊 {block_title}")

                if block_type == "ascii_bar":
                    lines = block.get("lines", [])
                    if lines:
                        bar_text = "\n".join(lines)
                        st.code(bar_text, language=None)
                elif block_type == "timeline":
                    items = block.get("items", [])
                    if items:
                        for item in items:
                            year = item.get("year", item.get("label", ""))
                            label = item.get("label", item.get("text", ""))
                            st.markdown(f"- **{year}** — {label}")
                elif block_type == "decision_matrix":
                    items = block.get("items", [])
                    if items:
                        import pandas as pd
                        df = pd.DataFrame(items)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    lines = block.get("lines", [])
                    if lines:
                        for line in lines:
                            st.markdown(line)

        # === 6. Disclaimer ===
        disclaimer = detail.get("disclaimer", "")
        if disclaimer:
            st.divider()
            st.caption(f"⚠️ {disclaimer}")

