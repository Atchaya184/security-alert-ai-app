// script.js -----------------------------------------------------
// Shared client-side logic: manual alert prediction, evidence
// rendering, Accept/Reject/Override (with mandatory reason), and
// high-impact action confirmation.
// -----------------------------------------------------------------

function showError(el, message) {
  if (!el) return;
  el.textContent = message;
  el.classList.add("show");
}

// ---------------------------------------------------------------
// Analyst auth token (rollback / high-impact-confirm actions only)
// ---------------------------------------------------------------
// This app has no login system; protected routes (model/rule rollback,
// high-impact action confirmation) require a shared analyst token instead
// (see utils/auth.py). We ask for it once per browser session and reuse it.
function getAuthToken() {
  let token = sessionStorage.getItem("analystAuthToken");
  if (!token) {
    token = window.prompt("Enter your analyst auth token to perform this action:", "") || "";
    if (token) sessionStorage.setItem("analystAuthToken", token);
  }
  return token;
}

function authHeaders(extra) {
  return Object.assign({ "X-Auth-Token": getAuthToken() }, extra || {});
}

// If a protected call comes back 401, the stored token was rejected --
// clear it so the next attempt re-prompts instead of retrying forever.
function forgetAuthTokenIfUnauthorized(status) {
  if (status === 401) sessionStorage.removeItem("analystAuthToken");
}

function hideError(el) {
  if (!el) return;
  el.classList.remove("show");
  el.textContent = "";
}

function colorForRecommendation(rec) {
  if (rec === "LIKELY TRUE INCIDENT") return "red";
  if (rec === "LIKELY FALSE POSITIVE") return "gray";
  return "purple"; // NOVEL / UNKNOWN
}

function shortRecLabel(rec) {
  if (rec === "LIKELY TRUE INCIDENT") return "Likely True Incident";
  if (rec === "LIKELY FALSE POSITIVE") return "Likely False Positive";
  return "Novel / Unknown";
}

// Simple analyst-facing label: Fake / Investigate / Escalate.
function simpleLabelFor(rec) {
  if (rec === "LIKELY TRUE INCIDENT") return "Escalate";
  if (rec === "LIKELY FALSE POSITIVE") return "Fake";
  return "Investigate";
}

function colorForRisk(level) {
  if (level === "High") return "red";
  if (level === "Medium") return "amber";
  if (level === "Low") return "green";
  return "gray";
}

// ---------------------------------------------------------------
// Manual alert form submission
// ---------------------------------------------------------------
async function submitAlertForm(formEl, resultEl, errorEl) {
  hideError(errorEl);
  resultEl.classList.remove("show");

  const formData = new FormData(formEl);
  const payload = Object.fromEntries(formData.entries());

  const submitBtn = formEl.querySelector("button[type='submit']");
  const originalText = submitBtn.innerHTML;
  submitBtn.disabled = true;
  submitBtn.innerHTML = '<span class="spinner"></span> Analyzing...';

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      showError(errorEl, data.error || "Something went wrong.");
      return;
    }

    renderPredictionResult(resultEl, data);
  } catch (err) {
    showError(errorEl, "Network error: could not reach the server.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = originalText;
  }
}

function renderPredictionResult(resultEl, data) {
  const color = data.color || colorForRecommendation(data.recommendation);
  const riskColor = data.risk_color || colorForRisk(data.risk_level);
  const simpleLabel = data.simple_label || simpleLabelFor(data.recommendation);

  let warningsHtml = "";
  if (data.high_risk_gate_triggered || data.requires_manual_confirmation) {
    const reasons = (data.high_risk_reasons || []).join(", ");
    warningsHtml += `
      <div class="high-impact-banner" style="border-color: rgba(255, 92, 114, 0.4); background: rgba(255, 92, 114, 0.08);">
        <div class="title" style="color: var(--red);">⚠ High-risk alert — manual confirmation required</div>
        <div style="font-size:0.85rem;color:var(--text-secondary);">${reasons ? `Reasons: ${reasons}. ` : ""}Quick Accept is disabled for this alert — choose Reject or Override with a reason, or resolve it from the Investigation page.</div>
      </div>`;
  }
  if (data.low_confidence) {
    warningsHtml += `
      <div class="high-impact-banner" style="border-color: rgba(255, 193, 7, 0.4); background: rgba(255, 193, 7, 0.08);">
        <div class="title">⚠ Low confidence prediction</div>
        <div style="font-size:0.85rem;color:var(--text-secondary);">Confidence is below the reliability threshold — treat this recommendation as a starting point and verify manually.</div>
      </div>`;
  }

  const acceptDisabled = data.high_risk_gate_triggered
    ? 'disabled title="High-risk alert — manual confirmation required. Use Reject or Override."'
    : "";

  let probsHtml = "";
  for (const [cls, val] of Object.entries(data.ml_probabilities || {})) {
    const barColor = cls === "True Incident" ? "red" : "gray";
    probsHtml += `
      <div style="margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:var(--text-secondary);">
          <span>P(${cls})</span><span>${val}%</span>
        </div>
        <div class="prob-bar">
          <div class="prob-bar-fill" style="width:${val}%;background:var(--${barColor});"></div>
        </div>
      </div>`;
  }

  let evidenceHtml = "";
  (data.evidence || []).forEach((e) => {
    evidenceHtml += `<li>${e}</li>`;
  });

  let highImpactHtml = "";
  if (data.is_high_impact_action) {
    highImpactHtml = `
      <div class="high-impact-banner">
        <div class="title">⚠ HIGH-IMPACT ACTION</div>
        <div style="font-size:0.9rem;">Recommended Action: <strong>${data.recommended_action}</strong></div>
        <div style="font-size:0.82rem;color:var(--text-secondary);margin-top:4px;">Simulated only — nothing is executed automatically. Human confirmation required.</div>
        <div class="action-group" style="margin-top:10px;">
          <button class="btn btn-accept btn-sm" onclick="confirmAction(${data.id}, 'Approved', this)">Approve</button>
          <button class="btn btn-reject btn-sm" onclick="confirmAction(${data.id}, 'Rejected', this)">Reject</button>
        </div>
      </div>`;
  }

  resultEl.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:14px;">
      <div>
        <div style="margin-bottom:6px;">
          <span class="badge ${color}"><span class="badge-dot"></span>${shortRecLabel(data.recommendation)}</span>
          <span class="badge ${riskColor}"><span class="badge-dot"></span>Risk: ${data.risk_level || "—"}</span>
        </div>
        <div style="font-size:0.85rem;font-weight:700;color:var(--text-secondary);">Recommendation: ${simpleLabel}</div>
        <div style="color:var(--text-secondary);font-size:0.85rem;margin-top:6px;">Confidence: ${data.confidence}%${data.low_confidence ? " ⚠" : ""} · Novelty score: ${data.novelty_score}</div>
        <div style="color:var(--text-secondary);font-size:0.75rem;margin-top:2px;font-family:monospace;">Rule: ${data.rule_triggered || "—"}</div>
      </div>
      <div class="action-group">
        <button class="btn btn-accept btn-sm" ${acceptDisabled} onclick="sendDecision(${data.id}, 'Accept', this)">Accept</button>
        <button class="btn btn-reject btn-sm" onclick="sendDecision(${data.id}, 'Reject', this)">Reject</button>
        <button class="btn btn-override btn-sm" onclick="openOverrideModal(${data.id})">Override</button>
      </div>
    </div>
    ${warningsHtml}
    <p style="color:var(--text-secondary);font-size:0.9rem;line-height:1.5;margin-top:12px;">${data.explanation}</p>
    <div style="margin-top:10px;">
      <div class="section-title" style="font-size:0.9rem;">Evidence</div>
      <ul class="evidence-list">${evidenceHtml}</ul>
    </div>
    <div style="margin-top:14px;">${probsHtml}</div>
    ${highImpactHtml}
    <div style="margin-top:12px;"><a href="/investigation/${data.id}">View full investigation page →</a></div>
  `;
  resultEl.classList.add("show");
}

// ---------------------------------------------------------------
// Accept / Reject / Override (mandatory reason)
// ---------------------------------------------------------------
let _overrideTargetId = null;

function openOverrideModal(id) {
  _overrideTargetId = id;
  const modal = document.getElementById("override-modal");
  if (!modal) return;
  document.getElementById("override-reason-input").value = "";
  document.getElementById("override-decision-select").value = "Likely False Positive";
  modal.classList.add("show");
}

function closeOverrideModal() {
  const modal = document.getElementById("override-modal");
  if (modal) modal.classList.remove("show");
  _overrideTargetId = null;
}

async function submitOverride() {
  const decisionSel = document.getElementById("override-decision-select");
  const reasonInput = document.getElementById("override-reason-input");
  const errorEl = document.getElementById("override-error");
  hideError(errorEl);

  const overrideDecision = decisionSel.value;
  const overrideReason = reasonInput.value.trim();

  if (!overrideReason) {
    showError(errorEl, "An override reason is required.");
    return;
  }

  try {
    const res = await fetch(`/decision/${_overrideTargetId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision: "Override", override_decision: overrideDecision, override_reason: overrideReason }),
    });
    const data = await res.json();
    if (!res.ok) {
      showError(errorEl, data.error || "Failed to record override.");
      return;
    }
    closeOverrideModal();
    location.reload();
  } catch (err) {
    showError(errorEl, "Network error while recording override.");
  }
}

async function sendDecision(id, decision, btn) {
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';
  try {
    const res = await fetch(`/decision/${id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision }),
    });
    const data = await res.json();
    if (res.ok) {
      btn.innerHTML = "✓ " + decision;
    } else {
      alert(data.error || "Failed to record decision.");
      btn.innerHTML = original;
    }
  } catch (err) {
    alert("Network error while recording decision.");
    btn.innerHTML = original;
  } finally {
    btn.disabled = false;
  }
}

async function rollbackDecision(id, btn) {
  btn.disabled = true;
  try {
    const res = await fetch(`/rollback/${id}`, { method: "POST" });
    const data = await res.json();
    if (res.ok) {
      alert(`Reverted to previous decision: ${data.reverted_to}`);
      location.reload();
    } else {
      alert(data.error || "Nothing to roll back to.");
    }
  } catch (err) {
    alert("Network error during rollback.");
  } finally {
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------
// High-impact action confirmation
// ---------------------------------------------------------------
async function confirmAction(id, confirmation, btn) {
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';
  try {
    const res = await fetch(`/confirm-action/${id}`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ confirmation }),
    });
    const data = await res.json();
    if (res.ok) {
      btn.innerHTML = "✓ " + confirmation;
    } else {
      forgetAuthTokenIfUnauthorized(res.status);
      alert(data.error || "Failed to record confirmation.");
      btn.innerHTML = original;
    }
  } catch (err) {
    alert("Network error while confirming action.");
    btn.innerHTML = original;
  } finally {
    btn.disabled = false;
  }
}
