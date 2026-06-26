const message = document.querySelector("#message");
const send = document.querySelector("#send");
const newSession = document.querySelector("#newSession");
const loadSession = document.querySelector("#loadSession");
const sessionId = document.querySelector("#sessionId");
const finalReport = document.querySelector("#finalReport");
const sessionPanel = document.querySelector("#sessionPanel");
const statusGrid = document.querySelector("#statusGrid");
const evidenceList = document.querySelector("#evidenceList");
const trace = document.querySelector("#trace");

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    message.value = button.dataset.example;
  });
});

newSession.addEventListener("click", () => {
  sessionId.value = `demo-${Date.now()}`;
  finalReport.textContent = "已新建会话，等待运行";
  renderSession({});
  renderEvidence([]);
  renderTrace([]);
});

loadSession.addEventListener("click", async () => {
  await refreshSession();
});

send.addEventListener("click", async () => {
  send.disabled = true;
  finalReport.textContent = "运行中...";
  try {
    const response = await fetch("/api/turns", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId.value, message: message.value }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || `HTTP ${response.status}`);
    }
    renderResult(data);
  } catch (error) {
    finalReport.textContent = `请求失败：${error.message}`;
  } finally {
    send.disabled = false;
  }
});

async function refreshSession() {
  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId.value)}`);
    const data = await response.json();
    renderSession(data);
  } catch (error) {
    finalReport.textContent = `状态刷新失败：${error.message}`;
  }
}

function renderResult(data) {
  // 兼容新旧格式
  const report = data.final_report;
  if (typeof report === "object" && report !== null) {
    finalReport.textContent = report.report_text || JSON.stringify(report, null, 2);
  } else {
    finalReport.textContent = report || "";
  }

  renderSession(data.session || {});
  renderStatus(data);
  renderEvidence(data.retrieved_evidence || (data.domain_rag_result && data.domain_rag_result.evidence) || []);
  renderTrace(data.trace || []);
}

function renderStatus(data) {
  // 新架构：consultation_request 包含意图信息
  const req = data.consultation_request || {};
  const evaluation = data.evaluation || {};
  const plan = data.execution_plan || {};

  const metrics = [
    ["Turn Type", data.turn_type || req.turn_type],
    ["Consultation Type", req.consultation_type || data.intent],
    ["Consultation Intent", req.consultation_intent || "-"],
    ["Risk Level", data.risk_level || req.risk_level || "-"],
    ["Execution Mode", plan.execution_mode || "-"],
    ["Active Nodes", (plan.active_nodes || data.selected_nodes || []).join(", ") || "-"],
  ];

  // 5 维评分
  if (evaluation.overall_score) {
    metrics.push(
      ["Evidence Score", evaluation.evidence_score?.toFixed(1) || "-"],
      ["Safety Score", evaluation.safety_score?.toFixed(1) || "-"],
      ["Practicality Score", evaluation.practicality_score?.toFixed(1) || "-"],
      ["Balance Score", evaluation.balance_score?.toFixed(1) || "-"],
      ["Actionability Score", evaluation.actionability_score?.toFixed(1) || "-"],
      ["Overall Score", evaluation.overall_score?.toFixed(1) || "-"],
      ["Need Revision", evaluation.need_revision ? "Yes" : "No"],
      ["Patch Target", evaluation.patch_target || "-"]
    );
  } else {
    metrics.push(["Critic", JSON.stringify(data.critic_result || {}, null, 2)]);
  }

  // 符号计算结果
  const sym = data.symbolic_result || {};
  const symResult = sym.result || {};
  if (symResult.day_master) {
    metrics.push(
      ["日主", `${symResult.day_master}（${symResult.day_master_strength}）`],
      ["喜用神", (symResult.favorable_elements || []).join(", ") || "-"],
      ["当前大运", symResult.current_cycle ? `${symResult.current_cycle.heavenly_stem || ""}${symResult.current_cycle.earthly_branch || ""}` : "-"],
      ["流年", symResult.current_year ? `${symResult.current_year.heavenly_stem || ""}${symResult.current_year.earthly_branch || ""}` : "-"]
    );
  }

  // 辩论结果
  const debate = data.debate_output || {};
  if (debate.debate_occurred !== undefined) {
    metrics.push(
      ["Debate Occurred", debate.debate_occurred ? "Yes" : "No"],
      ["Debate Rounds", debate.rounds_taken || 0]
    );
  }

  // 综合结果
  const synthesis = data.synthesis || {};
  if (synthesis.confidence) {
    metrics.push(
      ["Synthesis Confidence", synthesis.confidence?.toFixed(2) || "-"],
      ["Consistency Score", synthesis.consistency_score?.toFixed(2) || "-"]
    );
  }

  statusGrid.innerHTML = metrics.map(renderMetric).join("");
}

function renderSession(session) {
  const metrics = [
    ["Session", session.session_id || sessionId.value],
    ["Turn ID", session.turn_id ?? 0],
    ["Current Topic", session.current_topic || "无"],
    ["Last Intent", session.last_intent || "无"],
    ["Pending Flow", JSON.stringify(session.pending_flow || {}, null, 2)],
  ];

  // 新架构字段
  if (session.user_profile) {
    metrics.push(
      ["Knowledge Level", session.user_profile.knowledge_level || "-"],
      ["Preferred Style", session.user_profile.preferred_style || "-"]
    );
  }
  if (session.relevant_memories && session.relevant_memories.length) {
    metrics.push(["Memories", session.relevant_memories.length + " 条"]);
  }

  sessionPanel.innerHTML = metrics.map(renderMetric).join("");
}

function renderEvidence(items) {
  if (!items.length) {
    evidenceList.innerHTML = `<div class="empty">暂无检索证据</div>`;
    return;
  }
  evidenceList.innerHTML = items
    .map(
      (item) => `
        <article class="evidence">
          <div><strong>${escapeHtml(item.title || item.doc_id)}</strong><span>score ${escapeHtml(item.score ?? "")}</span></div>
          <p>${escapeHtml(item.text || "").slice(0, 180)}</p>
          <small>${escapeHtml(item.source_name || "")}</small>
        </article>
      `
    )
    .join("");
}

function renderTrace(events) {
  if (!events.length) {
    trace.innerHTML = `<div class="empty">暂无 Trace</div>`;
    return;
  }
  trace.innerHTML = events
    .map(
      (event) => `
        <div class="trace-item">
          <strong>${escapeHtml(event.node_name || "node")}</strong>
          <span>${escapeHtml(event.output_summary || "")}</span>
          <small>${escapeHtml(event.selected_reason || "")}</small>
        </div>
      `
    )
    .join("");
}

function renderMetric([label, value]) {
  return `<div class="metric"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(String(value ?? ""))}</span></div>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

refreshSession();
