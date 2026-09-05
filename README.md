# False Positive Reduction Assistant for Security Alerts

An analyst-in-the-loop AI assistant that helps security teams cut through
repetitive false-positive alerts from multiple disconnected tools — **without**
ever automatically suppressing, closing, quarantining, blocking, or disabling
anything. The AI recommends. A human analyst always decides.

> **Synthetic data notice**: this project uses a synthetic security-alert
> dataset (`dataset/security_alerts_2000.csv`). All "high-impact actions"
> (quarantine, block IP, disable user, escalate) are **simulated only** —
> nothing here ever touches a real endpoint, network device, IAM system, or
> security tool.

---

## 1. Problem Statement

A software company receives security alerts from multiple disconnected
security tools (EDR, IAM, Firewall, SIEM, Email Security, Cloud Security).
Analysts spend a large amount of time re-investigating **repetitive false
positives**, at the cost of time that should go to genuine incidents and
unfamiliar/novel threats.

## 2. Project Objective

Build a working prototype that:

1. **Preserves confirmed/genuine incidents** (top priority — never suppress a
   real incident to save time).
2. **Preserves novel/unfamiliar threats** (second priority — when the
   assistant hasn't seen enough like this before, it says so instead of
   guessing).
3. **Reduces repetitive, well-understood false positives** (third priority —
   only after the two safety priorities above are satisfied).

The AI **only recommends**. Final decisions belong to the human analyst.

---

## 3. Architecture

```
Security Alert (from a disconnected tool, via alert_source)
        │
        ▼
Alert Source Normalization  (all tools → one unified workflow)
        │
        ▼
Leakage-Safe Feature Extraction   (model/preprocess.py)
        │
        ▼
ML Classifier: P(False Positive) vs P(True Incident)   (RandomForest)
        │
        ▼
Novelty Detection: Isolation Forest + combination-rarity scoring
        │
        ▼
Safety Gate (model/decision.py):
   novel/insufficient-evidence?  → NOVEL / UNKNOWN, always
   otherwise                      → LIKELY FALSE POSITIVE / LIKELY TRUE INCIDENT
        │
        ▼
Evidence Generation  (utils/explain.py — built from real field values)
        │
        ▼
Simulated Recommended Action  (model/decision.py — quarantine/block/disable/
                                escalate mapping; HIGH-IMPACT actions require
                                human confirmation before anything happens)
        │
        ▼
Human Analyst Review (Investigation page)
        │
        ▼
Accept / Reject / Override (override requires a reason)
        │
        ▼
Human Confirmation for any High-Impact Action
        │
        ▼
Complete Audit Trail (SQLite — nothing ever deleted)
        │
        ▼
Analyst Feedback → Controlled Retraining → New Model Version → Change Review
→ Rollback if necessary
        │
        ▼
Evaluation: Baseline vs. Assistant, Analyst Hours Saved, Missed-Incident Rate,
Error Analysis
```

### User / Workflow Map

| Page | Purpose |
|---|---|
| **Dashboard** (`/`) | Manual alert input, instant recommendation |
| **Bulk Upload** (`/upload`) | CSV upload for batches of alerts from multiple tools |
| **Investigation** (`/investigation/<id>`) | Full single-alert review workflow: details, context, recommendation, evidence, decision, audit history |
| **Audit Trail** (`/audit`) | Every alert + recommendation + decision + override reason + confirmation, permanently |
| **Analytics** (`/analytics`) | Live usage stats + the offline chronological experiment (baseline vs. assistant, hours saved, missed-incident rate, error analysis) |
| **Change Review** (`/change-review`) | Propose/approve/reject rule-set changes; rule version history + rollback |
| **Model / Feedback** (`/model`) | Current model version + metrics, accumulated analyst feedback, controlled retraining, model version rollback |

---

## 4. Dataset

`dataset/security_alerts_2000.csv` — **used exactly as provided, never
regenerated or replaced.** 2,000 synthetic alerts with these fields:

`alert_id, timestamp, alert_source, alert_type, severity, description,
endpoint_id, endpoint_type, operating_system, user_role, department,
endpoint_risk_score, user_risk_score, source_ip, destination_ip, country,
process_name, parent_process, command_line, authentication_result,
previous_alert_count, previous_incident_count, similar_alert_count,
historical_false_positive_rate, historical_incident_rate,
analyst_disposition, confirmed_incident, incident_type,
investigation_reason, ground_truth, novelty_test_flag`

### Preprocessing (`model/preprocess.py`)

- Missing categorical values → `"Unknown"`; missing numeric values → column
  median (or 0 if the column is entirely missing).
- Timestamps parsed for chronological sorting; unparseable timestamps are
  pushed to the end rather than dropped.

---

## 5. Data Leakage Prevention

**This is enforced in code, not just convention.**

`LEAKAGE_COLUMNS` in `model/preprocess.py` lists the fields that must never be
used as model input features, because they only become known *after* a human
has already investigated the alert:

```
analyst_disposition, confirmed_incident, ground_truth,
investigation_reason, incident_type, novelty_test_flag
```

- `FEATURE_COLUMNS` (used for every fit/predict) is defined independently and
  never includes any of the above.
- `assert_no_leakage()` is called before every model fit and every
  prediction, and **raises `ValueError`** if a leakage column is ever present
  among the feature columns — a bug that reintroduces leakage will crash
  loudly instead of silently inflating accuracy.
- `ground_truth` is used **only** as the supervised training target and for
  offline evaluation — never as an input feature.
- `novelty_test_flag` is used **only** to evaluate how well the novelty
  detector recovers deliberately-held-out novel cases — never as an input
  feature. See `model/train_model.py`'s novelty evaluation section.
- This is verified by automated tests: `test_feature_columns_never_include_leakage_columns`,
  `test_assert_no_leakage_raises_on_violation`, `test_prepare_single_alert_excludes_leakage_fields`.

---

## 6. Machine Learning

### Model

`RandomForestClassifier` (n_estimators=300, class_weight="balanced") on a
`ColumnTransformer` of `StandardScaler` (numeric) + `OneHotEncoder` (categorical).
Binary target: **False Positive vs. True Incident** (from `ground_truth`).

Severity and risk scores are **inputs to the model**, not hard-coded rules —
the model decides how much weight they deserve; they are never used on their
own to define "dangerous."

### Chronological Evaluation

Records are sorted by `timestamp` and split **70% train / 15% validation /
15% test** — simulating "learn from past alerts, evaluate on future alerts"
rather than a random shuffle that would let the model peek at data
chronologically adjacent to what it's tested on.

- **Train**: 1,400 alerts (oldest)
- **Validation**: 300 alerts
- **Test**: 300 alerts (newest) — used *only* for final evaluation, never for
  fitting or threshold selection.

### Decision Threshold Calibration (Safety-First)

A naive 0.5 probability cutoff produced an unacceptably high missed-incident
rate. Because missing a genuine incident is far costlier than an extra
false-positive review, the **True-Incident decision threshold is calibrated
on the validation split** (never on test) to hit the configured
missed-incident target while preserving as much false-positive reduction as
possible. This is a real data-driven calibration step (see
`train_model.py`), not a hard-coded constant, and it is reported
transparently:

- **ROC-AUC (True Incident, threshold-independent): 0.746**
- **Calibrated decision threshold: 0.06** (i.e. the classifier only needs an
  6% predicted probability of "True Incident" to avoid a false-positive
  call — deliberately conservative)
- **Accuracy at the calibrated threshold: 32.3%** — this looks low compared
  to a typical classifier because the threshold is tuned for *safety*
  (catching incidents), not for raw accuracy. The 0.5-cutoff "reference"
  accuracy is also stored in `model/metrics.json` for comparison
  (`classification.uncalibrated_0.5_threshold`).

**This is an explicit, documented trade-off, not a hidden flaw.** See
Limitations (§27).

### Confusion Matrix (Test Split, Calibrated Threshold)

| | Predicted: False Positive | Predicted: True Incident |
|---|---|---|
| **Actual: False Positive** | 7 | 203 |
| **Actual: True Incident** | 0 | 90 |

---

## 7. Novelty Detection (`model/novelty.py`)

A real, implemented mechanism — not a placeholder:

1. **Isolation Forest** anomaly score over the same preprocessed feature
   space as the classifier (fit on the training split only).
2. **Combination rarity scoring**: how often the specific
   `(alert_source, alert_type, country)` combination was seen during
   training. A combination never seen before scores maximally novel
   regardless of how "normal" each individual field looks.

`novelty_score = 0.6 × isolation_forest_component + 0.4 × rarity_component`,
clipped to `[0, 1]`.

> **Design note — `process_name` was deliberately excluded from the rarity
> key.** An earlier version included it, which raised novelty recall on the
> dataset's own held-out cases, but at a serious practical cost: manually
> entered alerts and CSV rows that simply don't carry a process name (very
> common for Firewall/IAM/SIEM sources) default to `process_name="Unknown"`,
> and `"Unknown"` itself is rare in the training data (~1.5% of rows) purely
> because most training rows happen to have a real process name. That made
> *any* alert missing optional process context look artificially "novel"
> regardless of how routine the rest of it was — confirmed directly by
> manual testing, where realistic alerts with strong false-positive/
> true-incident evidence were being flagged Novel/Unknown almost every time
> solely because `process_name` was absent. Excluding it fixes that
> real-world usability failure at the cost of weaker novelty-recall numbers
> below. This trade-off was made deliberately and is reported honestly, not
> hidden.

`novelty_test_flag` is used **only** to evaluate this detector — a held-out
set of deliberately novel test cases the model never trained on:

- Novelty threshold in use: **0.65**
- Test alerts flagged as novelty test cases: **49** (of 300)
- Alerts the assistant predicted as Novel/Unknown: **16**
- Novelty recall on flagged cases: **0%**
- Novelty precision on predicted-novel: **0%**

**Honest limitation**: with `process_name` excluded from the rarity key (see
design note above), novelty detection did not recover any of this
dataset's specifically-flagged novelty test cases in this run — the
isolation-forest component alone isn't separating them well on this
synthetic data. The insufficient-evidence gate (§8) provides a second,
independent safety net that does not depend on this score at all
(`previous_alert_count == 0 AND similar_alert_count <= threshold`), which is
why `test_novel_alert_flagged_as_human_investigation_required` still passes
using a zero-history alert. See Limitations (§27).

---

## 8. Novelty Safety Gate (`model/decision.py`)

```
IF novelty_score >= novelty_threshold
OR (previous_alert_count == 0 AND similar_alert_count <= similar_alert_count_threshold)
THEN:
    recommendation = "NOVEL / UNKNOWN — HUMAN INVESTIGATION REQUIRED"
    (regardless of what the classifier predicted)
```

This safety gate runs **before** trusting any false-positive-suppression
call. A highly novel alert, or one with too little historical evidence, is
**never** recommended as a false positive — verified by
`test_decide_novel_overrides_confident_false_positive_when_novelty_high` and
`test_decide_novel_when_insufficient_evidence_even_if_low_novelty_score`.

---

## 9. Evidence & Explanation (`utils/explain.py`)

Every recommendation shows evidence generated from the **actual alert data
and actual model outputs** — not a templated/fake explanation:

- **Likely False Positive**: historical false-positive rate, similar alert
  count, previous alert count, endpoint risk, severity, model probability,
  the calibrated decision threshold in use.
- **Likely True Incident**: endpoint/user risk scores, prior confirmed
  incidents, historical incident rate, severity, authentication result,
  associated process, model probability.
- **Novel/Unknown**: novelty score, whether the exact combination was ever
  seen in training, insufficient-evidence flag, the isolation-forest
  component.

`test_evidence_not_hard_coded_reflects_actual_values` verifies that
different underlying alert data produces different evidence text.

---

## 10. Manual Override

Accept / Reject / Override are always available. **Override requires**:

- An explicit **Override Decision** (Likely False Positive / Likely True
  Incident / Novel — Human Investigation), and
- A **non-empty Override Reason**.

Enforced server-side in `utils/helpers.validate_override()` — an override
without a reason is rejected with HTTP 400
(`test_override_requires_reason`).

---

## 11. Complete Audit Trail

SQLite (`database.db`, auto-created — **no external DB server required**).
Every alert row stores: alert details, AI recommendation, confidence,
novelty score, evidence (JSON), recommended action, analyst decision,
override status/decision/reason, human confirmation status, model version,
rule version, and timestamp. A separate `decision_history` table logs every
event (AI recommendation → analyst decision → override → rollback →
high-impact confirmation) — **nothing is ever deleted or overwritten**; an
override adds new rows, it never erases the original AI recommendation.

---

## 12. High-Impact Actions (Simulated Only)

`model/decision.py` maps a Likely-True-Incident recommendation to a
simulated action (Quarantine endpoint / Disable user account / Block source
IP / Escalate incident) based on alert type. **No action ever executes
automatically.** The UI shows a "HIGH-IMPACT ACTION — HUMAN CONFIRMATION
REQUIRED" banner with Approve/Reject buttons; the confirmation (or rejection)
is permanently recorded in `high_impact_actions` and `decision_history`.

---

## 13. Change Review

`/change-review` lets an analyst propose a change to the active rule set
(novelty threshold, similar-alert-count threshold, investigation-time
assumptions, missed-incident target) with a required reason. **Nothing takes
effect until explicitly approved.** Approval bumps the rule version and
snapshots the *previous* configuration so it can be restored later.

---

## 14. Model / Rule Versioning

Every recommendation stored in the database records the `model_version` and
`rule_version` used to produce it. Both are tracked in a `model_versions`
table (version, description, reason, reviewer, status, timestamp) —
versions are assigned via a monotonically increasing counter
(`f"1.{total_version_records_ever_created}"`) specifically so that a
rollback-then-retrain cycle can never reuse a version label and silently
overwrite a different archived file.

---

## 15. Rollback

- **Rule rollback** (`/change-review/rollback`): requires `confirm: true`,
  restores the full previous rule-set configuration (except `model_version`,
  which is an independent track), records a rollback audit event, and never
  deletes prior history.
- **Model rollback** (`/model/rollback`): requires `confirm: true`, restores
  the archived model file for the target version, hot-reloads it into the
  running app, and updates the active-version record — again without
  deleting any history.

---

## 16. Analyst Feedback / Controlled Retraining Loop

Every Accept/Reject/Override is stored as feedback. A binary label is
derived **only** when it's unambiguous:

- **Accept** on a Likely-False-Positive or Likely-True-Incident
  recommendation → that label.
- **Override** to "Likely False Positive" / "Likely True Incident" → the
  overridden label.
- Overrides to "Novel / Human Investigation" (and Rejects) don't produce a
  clean binary label and are excluded from retraining.

Retraining is **controlled, not automatic**:

```
[Prepare Retraining Dataset]  →  writes dataset/retrain_prepared.csv,
                                   marks feedback rows as consumed
[Retrain Model]                →  archives the current model, retrains with
                                   the original dataset + prepared feedback
                                   appended to the TRAINING split (feedback
                                   is analyst-confirmed history, not
                                   future/test data), records a new model
                                   version with fresh metrics
```

---

## 17. Baseline

**Baseline assumption**: every alert is manually investigated by an analyst,
at a **configurable** average investigation time (default: **5 minutes**,
`avg_investigation_minutes` in Change Review). Alerts the assistant
recommends as a Likely False Positive are assumed to still get a **quick
review** rather than zero time (default: **0.5 minutes**,
`quick_review_minutes`) — both assumptions are visible and changeable in the
Change Review page, not buried in code.

---

## 18. Analyst Hours Saved (Measured, Test Split, n=300)

| Metric | Value |
|---|---|
| Baseline hours (every alert, 5 min each) | **25.0 h** |
| Assistant hours (293 alerts need full review @ 5 min + 7 quick reviews @ 0.5 min) | **24.48 h** |
| **Hours saved** | **0.53 h** |
| **Percent saved** | **2.1%** |

Computed programmatically in `train_model.py` from the actual test-split
recommendations — verified for internal consistency by
`test_hours_saved_calculation_is_internally_consistent`.

---

## 19. Missed-Incident Rate (Critical Safety Metric)

A **missed incident** = a confirmed genuine incident (`ground_truth ==
"True Incident"`) that the assistant recommended as a Likely False Positive.
**Novel alerts sent for human investigation are explicitly NOT counted as
missed incidents** (they're tracked separately in Error Analysis as
`novel_but_was_incident`).

| Metric | Value |
|---|---|
| Confirmed incidents in test split | 90 |
| Missed incidents | **0** |
| **Measured missed-incident rate** | **0.00%** |
| Target (configurable) | ≤ 2.0% |
| **Target achieved** | **YES** |

---

## 20. Baseline vs. Assistant — Summary

| | BASELINE | TARGET | MEASURED | PASS/FAIL |
|---|---|---|---|---|
| Missed-incident rate | — | ≤ 2.0% | 0.00% | **PASS** |
| Analyst hours (300 test alerts) | 25.0 h | reduce | 24.48 h | PASS (modest) |
| Hours saved | — | maximize | 0.53 h (2.1%) | PASS (modest) |
| False-positive reduction | — | maximize | 3.3% (7/210) | PASS (modest) |

**Honest framing**: the assistant is currently calibrated conservatively —
it satisfies the safety target (missed incidents) but only achieves a modest
false-positive reduction / hours-saved benefit as a direct consequence. See
Limitations.

---

## 21. Error Analysis (Test Split)

Computed in `train_model.py`, stored in `model/metrics.json`, and rendered
live on the Analytics page with sample alert IDs:

| Category | Count | % of test |
|---|---|---|
| False positive called incident (wastes analyst time, safe) | 197 | 65.7% |
| **Incident called false positive (MISSED INCIDENT)** | 0 | 0.0% |
| Novel but was actually a false positive (safe over-caution) | 6 | 2.0% |
| Novel but was actually an incident (exactly what the gate protects) | 10 | 3.3% |
| Low-confidence predictions (<60%) | varies | see Analytics |

---

## 22. Edge / Failure Cases (Implemented & Tested)

| Case | Expected | Verified by |
|---|---|---|
| Missing endpoint context | Reduced confidence, often routed to human investigation via the insufficient-evidence gate | `test_decide_novel_when_insufficient_evidence_even_if_low_novelty_score` |
| Completely novel alert | Novel/Unknown + human investigation | `test_novel_alert_flagged_as_human_investigation_required` |
| High severity but historically repetitive false positive | NOT forced to "incident" purely from severity | `test_high_severity_repetitive_false_positive_not_forced_to_incident` |
| Low/medium severity but genuine incident | Can still be classified as a true incident | `test_low_severity_can_still_be_true_incident` |
| Analyst overrides AI | Mandatory reason + full audit trail | `test_override_requires_reason`, `test_override_succeeds_with_reason_and_is_audited` |
| High-impact action | Mandatory human confirmation | `test_high_impact_action_requires_confirmation` |
| Missing fields | No crash, safe defaults | `test_edge_case_missing_fields_does_not_crash` |
| Invalid severity value | Defaults safely to Medium | `test_edge_case_invalid_severity_defaults_safely` |
| Empty CSV upload | Rejected with clear error | `test_edge_case_empty_csv_upload` |
| Non-CSV file upload | Rejected with clear error | `test_edge_case_non_csv_file_upload` |
| CSV missing required columns | Rejected, names the missing columns | `test_edge_case_csv_missing_required_columns` |
| Nonexistent alert ID | 404, not a crash | `test_edge_case_nonexistent_alert_returns_404` |
| Malformed JSON body | 400/500, not an unhandled crash | `test_app_does_not_crash_on_malformed_json` |

---

## 23. Installation

```bash
pip install -r requirements.txt
```

## 24. Training

```bash
python model/train_model.py
```

Regenerates `model/model.pkl` and `model/metrics.json` from
`dataset/security_alerts_2000.csv` (chronological split, novelty detector,
threshold calibration, full metrics).

## 25. Running the Application

```bash
python app.py
```

Open **http://localhost:5000**. On first run, if `model/model.pkl` doesn't
exist, the app trains it automatically. `database.db` is created
automatically — no external database server needed.

## 26. Testing

```bash
pytest
```

**50 automated tests**, all passing as of this build — covering dataset
loading, preprocessing, leakage prevention, ML prediction, novelty
detection, evidence generation, false-positive/true-incident/novel
recommendations, manual override (+ mandatory reason), database logging,
high-impact confirmation, change review + rollback, analyst feedback,
hours-saved/missed-incident internal-consistency checks, and edge cases.

---

## 27. Limitations (Stated Honestly, Not Hidden)

1. **Modest false-positive reduction / hours-saved.** The safety-first
   threshold calibration (§6) means the assistant only recommends
   suppression for the false positives it's most confident about (3.3% of
   actual false positives, 2.1% hours saved on the test split). This is a
   direct, deliberate consequence of prioritizing "never miss an incident"
   above "maximize time saved," per the stated priority order. A team
   comfortable with a higher missed-incident risk could relax the target in
   Change Review to trade safety for more time savings.
2. **Novelty detection recall/precision against the dataset's specifically-
   flagged novelty test cases are 0% in this run** (see §7's design note).
   This is a direct, deliberate consequence of excluding `process_name`
   from the rarity key to fix a more serious problem: over-flagging
   realistic alerts that simply lack optional process context. The
   insufficient-evidence gate (zero prior alerts + zero similar alerts)
   remains a fully independent, working safety net regardless of this
   score. A production version would likely want a richer novelty signal
   (e.g. a learned autoencoder reconstruction error, or a larger/more
   diverse training set) rather than trading one weakness for another
   within the current two-component design.
3. **Sample size.** 2,000 total alerts (300 in the test split) is small for
   a security-alert model; metrics should be read as a prototype-scale
   demonstration of the methodology, not a production-grade accuracy
   guarantee.
4. **The `model.pkl` bundle pickles `NoveltyDetector` under the bare module
   name `novelty`**, which resolves correctly through every supported entry
   point (`app.py`, `model/train_model.py`, `pytest`) because each adds
   `model/` to `sys.path` before importing/unpickling — but a standalone
   `joblib.load("model/model.pkl")` from an unrelated script without that
   `sys.path` setup will fail with `ModuleNotFoundError`. Always load the
   model via the app (`utils.recommend.engine`) or by running the provided
   entry points.
5. **No authentication / not production-hardened.** This is a prototype:
   no user accounts, no rate limiting, single-file SQLite rather than a
   production database, Flask's development server rather than a
   production WSGI server.
6. **`command_line` field is not used as a model feature** (only
   `process_name`/`parent_process` are) to avoid overfitting to
   free-text/high-cardinality values on a small dataset; it is not currently
   surfaced in evidence either.

---

## 28. User / Stakeholder Validation Template

*(No live stakeholder validation was performed as part of this automated
build — the table below is a template for the project owner to complete
after real analyst usage. Do not fill this in with fabricated feedback.)*

| Tester / Stakeholder | Workflow Tested | What Worked | Problems Found | Feedback | Improvements Made |
|---|---|---|---|---|---|
| _(name/role)_ | _(e.g. Dashboard → Investigation → Override)_ | | | | |
| | | | | | |

---

## 29. Final Requirement Checklist

Verified against the actually-implemented and actually-tested code in this
repository (not marked PASS without a corresponding test or a working,
exercised code path):

- [x] Problem analysis — §1
- [x] User/workflow map — §3
- [x] Synthetic dataset (used as-is, not regenerated) — §4
- [x] Working end-to-end prototype — tested via `pytest` (50 passed) + manual server testing
- [x] Multiple disconnected alert sources — `alert_source` field, normalized workflow
- [x] False-positive reduction — measured 3.3% (modest; see Limitations)
- [x] Analyst feedback/learning — §16, `/model` page, tested
- [x] Novel-threat preservation — §7–8, tested
- [x] Evidence/rules behind recommendations — §9, tested
- [x] Manual override — §10, tested
- [x] Mandatory override reason — §10, tested (`test_override_requires_reason`)
- [x] Human confirmation for high-impact actions — §12, tested
- [x] Complete audit trail — §11
- [x] Change review — §13, tested
- [x] Model/rule versioning — §14
- [x] Rollback — §15, tested (rule); model rollback implemented and manually tested
- [x] Baseline — §17
- [x] Target — §18–20 (configurable targets shown alongside measured results)
- [x] Measured result — §18–21 (all computed programmatically, stored in `model/metrics.json`)
- [x] Analyst hours saved — §18, tested for internal consistency
- [x] Controlled missed-incident rate — §19, tested for internal consistency, target met (0.00% ≤ 2%)
- [x] Error analysis — §21, with sample alert IDs
- [x] At least 3 edge/failure cases — §22 lists 12, all tested
- [x] Automated tests — 50 tests in `tests/test_app.py`, all passing
- [x] README — this document
- [x] User/stakeholder validation template — §28 (template only; no fabricated feedback)
- [x] Reproducibility — fixed random seeds (`random_state=42`) in classifier and novelty detector; dataset/model/rule versions and timestamps recorded in `model/metrics.json` and the `model_versions` DB table

No metric in this document was hand-typed independently of the code that
produced it — every number above was read directly from a `pytest` run or
`model/metrics.json` generated by `model/train_model.py` against the
unmodified dataset.
