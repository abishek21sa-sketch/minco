"""Self-contained executive Control Tower UI served by the governed API."""

from __future__ import annotations

CONTROL_TOWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MINCO Command Center</title>
  <style>
    :root {
      --navy: #07182f;
      --navy-2: #0d2748;
      --blue: #1f6feb;
      --cyan: #22b8cf;
      --ink: #10233f;
      --muted: #6f8098;
      --line: #dce5ef;
      --canvas: #f4f7fb;
      --panel: #ffffff;
      --green: #16845b;
      --amber: #ae7411;
      --red: #bf3d45;
      --shadow: 0 12px 32px rgba(16, 35, 63, .08);
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--canvas); color: var(--ink); font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }
    button, input { font: inherit; }
    button { cursor: pointer; }
    .shell { min-height: 100vh; display: grid; grid-template-columns: 250px 1fr; }
    .sidebar { background: var(--navy); color: #d7e5f5; padding: 24px 16px; display: flex; flex-direction: column; gap: 24px; }
    .brand { display: flex; align-items: center; gap: 11px; padding: 4px 10px 20px; border-bottom: 1px solid rgba(255,255,255,.13); }
    .brand-mark { width: 32px; height: 32px; display: grid; place-items: center; border-radius: 9px; color: white; font-weight: 850; background: linear-gradient(135deg, var(--blue), var(--cyan)); box-shadow: 0 7px 20px rgba(34,184,207,.25); }
    .brand-name { font-weight: 800; letter-spacing: .07em; font-size: 16px; }
    .brand-sub { display: block; color: #8fa8c2; font-size: 10px; letter-spacing: .12em; margin-top: 2px; }
    .nav-title { color: #6f8ba9; font-size: 10px; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; padding: 0 10px; }
    .nav { display: grid; gap: 5px; }
    .nav button { border: 0; text-align: left; color: #a9bdd3; background: transparent; padding: 11px 12px; border-radius: 8px; font-weight: 650; }
    .nav button:hover, .nav button.active { color: white; background: rgba(79, 157, 244, .18); }
    .nav button span { display: inline-block; width: 22px; color: #6dc9e1; }
    .sidebar-foot { margin-top: auto; border: 1px solid rgba(255,255,255,.12); border-radius: 10px; padding: 13px; color: #94abc3; font-size: 11px; }
    .sidebar-foot strong { color: white; display: block; margin-bottom: 4px; }
    .main { min-width: 0; }
    .topbar { height: 76px; background: var(--panel); border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; padding: 0 30px; gap: 18px; }
    .eyebrow { color: var(--blue); font-size: 11px; font-weight: 850; letter-spacing: .14em; text-transform: uppercase; }
    .topbar h1 { margin: 3px 0 0; font-size: 22px; letter-spacing: -.02em; }
    .top-actions { display: flex; align-items: center; gap: 9px; }
    .pill { display: inline-flex; align-items: center; gap: 7px; border: 1px solid var(--line); border-radius: 999px; padding: 7px 11px; font-size: 11px; font-weight: 750; color: var(--muted); background: #fbfdff; }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); }
    .dot.amber { background: #d8941c; }
    .ghost { background: white; color: var(--blue); border: 1px solid #c7d8ec; border-radius: 7px; padding: 8px 12px; font-weight: 750; }
    .ghost:hover { background: #eef6ff; }
    .content { padding: 26px 30px 42px; max-width: 1760px; }
    .hero { display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; margin-bottom: 22px; }
    .hero h2 { margin: 0; font-size: 28px; letter-spacing: -.035em; }
    .hero p { margin: 5px 0 0; color: var(--muted); }
    .hero-meta { text-align: right; color: var(--muted); font-size: 12px; }
    .hero-meta strong { color: var(--ink); display: block; font-size: 13px; }
    .error { display: none; background: #fff1f1; border: 1px solid #f1b7bb; color: #8e2730; border-radius: 10px; padding: 13px 15px; margin-bottom: 18px; }
    .kpis { display: grid; grid-template-columns: repeat(5, minmax(150px, 1fr)); gap: 13px; margin-bottom: 18px; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 13px; box-shadow: var(--shadow); }
    .kpi { padding: 17px 18px; min-height: 112px; }
    .kpi-label { color: var(--muted); font-size: 11px; font-weight: 750; text-transform: uppercase; letter-spacing: .09em; }
    .kpi-value { font-size: 25px; font-weight: 850; margin-top: 12px; letter-spacing: -.03em; }
    .kpi-note { color: var(--muted); font-size: 11px; margin-top: 3px; }
    .green { color: var(--green); } .amber { color: var(--amber); } .red { color: var(--red); } .blue { color: var(--blue); }
    .grid { display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(330px, .8fr); gap: 18px; margin-bottom: 18px; }
    .panel { padding: 20px; }
    .panel-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 16px; }
    .panel-title { font-size: 16px; font-weight: 850; margin: 0; }
    .panel-sub { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .section { scroll-margin-top: 90px; }
    .status-banner { display: flex; align-items: center; justify-content: space-between; gap: 14px; background: linear-gradient(115deg, #0b315a, #0f5b79); color: white; border-radius: 10px; padding: 16px 18px; margin-bottom: 16px; }
    .status-banner small { color: #b9e7f1; text-transform: uppercase; font-size: 10px; letter-spacing: .12em; font-weight: 800; }
    .status-banner strong { display: block; margin-top: 3px; font-size: 20px; letter-spacing: .03em; }
    .status-tag { border: 1px solid rgba(255,255,255,.35); border-radius: 999px; padding: 7px 11px; color: #e8fbff; font-size: 11px; font-weight: 800; }
    .metric-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .mini { border: 1px solid var(--line); border-radius: 9px; padding: 13px; background: #fbfdff; }
    .mini label { display: block; color: var(--muted); font-size: 11px; }
    .mini strong { display: block; margin-top: 7px; font-size: 20px; }
    .list { display: grid; gap: 2px; }
    .list-row { display: flex; justify-content: space-between; gap: 14px; padding: 12px 0; border-bottom: 1px solid #edf1f5; }
    .list-row:last-child { border-bottom: 0; }
    .list-row strong { font-size: 13px; }
    .list-row span { color: var(--muted); font-size: 12px; text-align: right; }
    .alert { display: flex; gap: 10px; padding: 11px 0; border-bottom: 1px solid #edf1f5; }
    .alert:last-child { border-bottom: 0; }
    .alert-icon { width: 8px; height: 8px; border-radius: 50%; margin-top: 5px; background: var(--amber); flex: 0 0 auto; }
    .alert.red .alert-icon { background: var(--red); }
    .alert strong { display: block; font-size: 12px; }
    .alert span { color: var(--muted); font-size: 12px; }
    .actions { display: grid; gap: 8px; }
    .action { display: grid; grid-template-columns: 28px 1fr auto; align-items: center; gap: 10px; border: 1px solid var(--line); border-radius: 9px; padding: 11px 12px; }
    .action-num { width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center; background: #eaf3ff; color: var(--blue); font-size: 11px; font-weight: 850; }
    .action-name { font-weight: 750; font-size: 12px; }
    .action-detail { color: var(--muted); font-size: 11px; display: block; margin-top: 2px; }
    .action-qty { color: var(--blue); font-size: 12px; font-weight: 850; }
    .posture-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
    .posture { border: 1px solid var(--line); border-radius: 10px; padding: 13px; }
    .posture-top { display: flex; justify-content: space-between; gap: 8px; }
    .posture strong { font-size: 12px; }
    .badge { font-size: 10px; border-radius: 999px; padding: 3px 7px; font-weight: 850; background: #eaf8f1; color: var(--green); }
    .badge.warn { color: var(--amber); background: #fff7e7; }
    .badge.block { color: var(--red); background: #fff0f1; }
    .posture p { margin: 8px 0 0; color: var(--muted); font-size: 11px; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { text-align: left; padding: 11px 9px; border-bottom: 1px solid #edf1f5; white-space: nowrap; }
    th { color: var(--muted); font-size: 10px; letter-spacing: .08em; text-transform: uppercase; }
    td strong { font-size: 12px; }
    .scenario-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .scenario { border: 1px solid var(--line); border-radius: 10px; padding: 13px; }
    .scenario strong { display: block; font-size: 13px; }
    .scenario p { color: var(--muted); font-size: 11px; margin: 5px 0 10px; min-height: 31px; }
    .chips { display: flex; flex-wrap: wrap; gap: 5px; }
    .chip { background: #f0f5fa; color: #55708d; border-radius: 5px; padding: 4px 6px; font-size: 10px; }
    .claim { border-left: 3px solid var(--cyan); background: #f0fbfd; color: #496c78; border-radius: 4px; padding: 12px 13px; font-size: 12px; margin-top: 14px; }
    .workbench-grid { display: grid; grid-template-columns: minmax(260px, .75fr) minmax(0, 1.25fr); gap: 16px; }
    .workbench-controls { border: 1px solid var(--line); border-radius: 10px; padding: 15px; background: #fbfdff; }
    .field { margin-bottom: 12px; }
    .field label { display: block; color: var(--muted); font-size: 11px; font-weight: 750; margin-bottom: 5px; }
    .field select, .field input, .field textarea { width: 100%; border: 1px solid #cbd9e7; border-radius: 7px; background: white; color: var(--ink); padding: 9px 10px; }
    .field textarea { min-height: 76px; resize: vertical; }
    .field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
    .primary { width: 100%; border: 0; border-radius: 7px; padding: 11px 13px; background: linear-gradient(110deg, var(--blue), #168da8); color: white; font-weight: 850; }
    .primary:hover { filter: brightness(1.06); }
    .primary:disabled { cursor: wait; opacity: .55; }
    .subtle-note { color: var(--muted); font-size: 11px; margin: 11px 0 0; }
    .comparison { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
    .comparison-head { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; background: #f2f6fb; padding: 11px 13px; color: var(--muted); font-size: 10px; font-weight: 850; letter-spacing: .08em; text-transform: uppercase; }
    .comparison-row { display: grid; grid-template-columns: 1.2fr 1fr 1fr 1fr; align-items: center; gap: 8px; padding: 10px 13px; border-top: 1px solid #edf1f5; font-size: 12px; }
    .comparison-row strong { font-size: 11px; }
    .comparison-row span { text-align: right; color: var(--muted); }
    .delta-good { color: var(--green) !important; font-weight: 800; }
    .delta-bad { color: var(--red) !important; font-weight: 800; }
    .review-box { margin-top: 16px; border-top: 1px solid var(--line); padding-top: 16px; }
    .review-grid { display: grid; grid-template-columns: 1fr 1fr 2fr auto; gap: 9px; align-items: end; }
    .review-grid .field { margin-bottom: 0; }
    .review-grid .primary { width: auto; white-space: nowrap; }
    .run-chip { background: #eef6ff; color: var(--blue); border-radius: 5px; padding: 4px 7px; font: 11px ui-monospace, SFMono-Regular, Consolas, monospace; }
    .review-history { margin-top: 13px; display: grid; gap: 6px; }
    .review-event { display: flex; justify-content: space-between; gap: 12px; padding: 9px 11px; border: 1px solid var(--line); border-radius: 8px; font-size: 11px; }
    .review-event span { color: var(--muted); }
    .review-status { color: var(--muted); font-size: 11px; min-height: 17px; margin-top: 9px; }
    .packet-row { display: flex; align-items: center; gap: 10px; margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--line); }
    .packet-row .ghost { width: auto; white-space: nowrap; }
    .packet-status { color: var(--muted); font-size: 11px; }
    .footer { color: var(--muted); text-align: center; font-size: 11px; padding: 9px; }
    .api-key { display: none; position: fixed; right: 25px; bottom: 25px; width: 290px; z-index: 5; padding: 14px; background: white; border: 1px solid var(--line); border-radius: 11px; box-shadow: 0 16px 50px rgba(7,24,47,.2); }
    .api-key.open { display: block; }
    .api-key label { display: block; color: var(--muted); font-size: 11px; margin-bottom: 6px; }
    .api-key input { width: 100%; border: 1px solid var(--line); border-radius: 7px; padding: 8px; }
    @media (max-width: 1150px) { .shell { grid-template-columns: 74px 1fr; } .brand-name, .brand-sub, .nav-title, .nav button:not(.active)::after, .sidebar-foot { display: none; } .brand { justify-content: center; padding-inline: 0; } .nav button { text-align: center; padding-inline: 5px; } .nav button span { width: auto; } .kpis { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 780px) { .shell { display: block; } .sidebar { display: none; } .topbar { padding: 0 16px; } .top-actions .pill { display: none; } .content { padding: 18px 14px 35px; } .hero { display: block; } .hero-meta { text-align: left; margin-top: 10px; } .kpis, .grid, .posture-grid, .scenario-grid, .workbench-grid, .review-grid { grid-template-columns: 1fr; } .metric-strip { grid-template-columns: repeat(2, 1fr); } .comparison-row { grid-template-columns: 1.2fr 1fr 1fr 1fr; font-size: 11px; } }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand"><div class="brand-mark">M</div><div><div class="brand-name">MINCO</div><span class="brand-sub">COMMAND CENTER</span></div></div>
      <div class="nav-title">Workspace</div>
      <nav class="nav">
        <button class="active" data-target="overview"><span>◈</span> Overview</button>
        <button data-target="workbench"><span>▣</span> Operations Workbench</button>
        <button data-target="decision"><span>↗</span> Decision runway</button>
        <button data-target="governance"><span>✓</span> Governance</button>
        <button data-target="history"><span>◌</span> Operational history</button>
        <button data-target="scenarios"><span>◇</span> Scenario catalog</button>
      </nav>
      <div class="nav-title">Controls</div>
      <nav class="nav"><button id="apiKeyButton"><span>⌁</span> API access</button><button id="refreshButton"><span>↻</span> Refresh snapshot</button></nav>
      <div class="sidebar-foot"><strong>Human-gated by design</strong>Recommendations are reviewable model outputs. This surface never dispatches clinical, staffing, transfer, or diversion actions.</div>
    </aside>
    <main class="main">
      <header class="topbar"><div><div class="eyebrow">FLOW-CVaR · Decision intelligence</div><h1>Hospital capacity operations</h1></div><div class="top-actions"><div class="pill"><i class="dot"></i> <span id="identityPill">API connected</span></div><div class="pill"><i class="dot amber"></i> Synthetic reference</div><button class="ghost" id="topRefresh">Refresh</button></div></header>
      <div class="content">
        <div id="error" class="error"></div>
        <section id="overview" class="section">
          <div class="hero"><div><h2>Executive operating picture</h2><p>One auditable view across network pressure, decision posture, evidence, and human review.</p></div><div class="hero-meta"><strong id="snapshotTime">Loading snapshot…</strong><span id="correlation">—</span></div></div>
          <div class="kpis">
            <div class="card kpi"><div class="kpi-label">Network posture</div><div class="kpi-value amber" id="kpiAlert">—</div><div class="kpi-note">Reference status signal</div></div>
            <div class="card kpi"><div class="kpi-label">Governance gate</div><div class="kpi-value red" id="kpiGovernance">—</div><div class="kpi-note">Human review boundary</div></div>
            <div class="card kpi"><div class="kpi-label">Projected census</div><div class="kpi-value blue" id="kpiCensus">—</div><div class="kpi-note" id="kpiHospitals">Across modeled hospitals</div></div>
            <div class="card kpi"><div class="kpi-label">CVaR loss</div><div class="kpi-value" id="kpiCvar">—</div><div class="kpi-note">Tail-risk objective</div></div>
            <div class="card kpi"><div class="kpi-label">Event history</div><div class="kpi-value" id="kpiEvents">—</div><div class="kpi-note" id="kpiFreshness">Replay posture</div></div>
          </div>
          <div class="grid">
            <section class="card panel"><div class="status-banner"><div><small>Control Tower status</small><strong id="towerState">Loading…</strong></div><div class="status-tag" id="towerMode">SYNTHETIC REFERENCE</div></div><div class="metric-strip"><div class="mini"><label>Modeled hospitals</label><strong id="modeledHospitals">—</strong></div><div class="mini"><label>Selected actions</label><strong id="selectedActions">—</strong></div><div class="mini"><label>Review state</label><strong id="reviewState">—</strong></div></div><div class="claim" id="claim">Loading evidence boundary…</div></section>
            <section class="card panel"><div class="panel-head"><div><h3 class="panel-title">Priority alerts</h3><div class="panel-sub">Signals requiring operator attention</div></div><span class="badge warn" id="alertCount">—</span></div><div id="alerts" class="list"><div class="panel-sub">Loading…</div></div></section>
          </div>
        </section>
        <section id="workbench" class="section card panel" style="margin-bottom:18px"><div class="panel-head"><div><h3 class="panel-title">Operations Workbench</h3><div class="panel-sub">Compare a governed counterfactual against baseline, then record a human disposition</div></div><span class="badge warn">SIMULATION · HUMAN REVIEW</span></div><div class="workbench-grid"><div class="workbench-controls"><div class="field"><label for="baselineScenario">Baseline scenario</label><select id="baselineScenario"></select></div><div class="field"><label for="counterfactualScenario">Counterfactual scenario</label><select id="counterfactualScenario"></select></div><div class="field-row"><div class="field"><label for="replications">Replications</label><input id="replications" type="number" min="1" max="200" value="10"></div><div class="field"><label>Execution mode</label><input value="Recommendation only" disabled></div></div><button id="runComparison" class="primary">Run governed comparison</button><div id="workbenchStatus" class="review-status">Select a scenario to begin. Evaluation writes an auditable run manifest but never dispatches an operational action.</div></div><div class="comparison"><div class="comparison-head"><span>Metric</span><span>Baseline</span><span>Counterfactual</span><span>Delta</span></div><div id="comparisonRows"><div class="panel-sub" style="padding:16px">No comparison run yet. Choose a counterfactual and run the model.</div></div></div></div><div class="review-box"><div class="panel-head" style="margin-bottom:10px"><div><h4 class="panel-title" style="font-size:14px">Human disposition</h4><div class="panel-sub">Review the selected run before recording an immutable decision event.</div></div><span id="reviewRunId" class="run-chip">No run selected</span></div><div class="review-grid"><div class="field"><label for="reviewerName">Reviewer identity</label><input id="reviewerName" placeholder="e.g. command-center-lead"></div><div class="field"><label for="reviewDecision">Disposition</label><select id="reviewDecision"><option value="ACCEPTED_FOR_OPERATIONS_REVIEW">Accept for operations review</option><option value="DEFERRED">Defer pending facts</option><option value="REJECTED">Reject recommendation</option></select></div><div class="field"><label for="reviewComment">Rationale</label><textarea id="reviewComment" placeholder="Record the human rationale and any facts still required."></textarea></div><button id="submitReview" class="primary" disabled>Record review</button></div><div id="reviewStatus" class="review-status"></div><div class="packet-row"><button id="packetButton" class="ghost" disabled>Build evidence packet</button><span id="packetStatus" class="packet-status">A packet bundles the run manifest, reviews, integrity, and release evidence.</span></div><div id="reviewHistoryRows" class="review-history"></div></div></section>
        <section id="decision" class="section card panel" style="margin-bottom:18px"><div class="panel-head"><div><h3 class="panel-title">Decision runway</h3><div class="panel-sub">Recommended decision objects from the governed FLOW-CVaR model</div></div><span class="badge block">RECOMMENDATION ONLY</span></div><div id="actions" class="actions"><div class="panel-sub">Loading decision actions…</div></div></section>
        <section id="governance" class="section card panel" style="margin-bottom:18px"><div class="panel-head"><div><h3 class="panel-title">Governance & evidence posture</h3><div class="panel-sub">Controls are visible together so review does not depend on hidden state</div></div><span class="badge block">NO AUTONOMOUS EXECUTION</span></div><div class="posture-grid"><div class="posture"><div class="posture-top"><strong>Audit integrity</strong><span class="badge" id="auditBadge">—</span></div><p id="auditText">Loading chain status…</p></div><div class="posture"><div class="posture-top"><strong>Release evidence</strong><span class="badge" id="evidenceBadge">—</span></div><p id="evidenceText">Loading artifact fingerprint…</p></div><div class="posture"><div class="posture-top"><strong>Live event gateway</strong><span class="badge warn" id="feedBadge">—</span></div><p id="feedText">Loading feed posture…</p></div><div class="posture"><div class="posture-top"><strong>Production solver</strong><span class="badge warn" id="solverBadge">—</span></div><p id="solverText">Loading dependency posture…</p></div></div><div class="claim" id="readinessText">Loading release-readiness boundary…</div></section>
        <div class="grid">
          <section id="history" class="section card panel"><div class="panel-head"><div><h3 class="panel-title">Operational history</h3><div class="panel-sub">Replay/event-lake provenance and freshness</div></div><span class="badge" id="historyBadge">—</span></div><div id="historyRows" class="list"></div></section>
          <section class="card panel"><div class="panel-head"><div><h3 class="panel-title">Network inventory</h3><div class="panel-sub">Observed reference-case operating footprint</div></div></div><div class="table-wrap"><table><thead><tr><th>Hospital</th><th>Projected census</th><th>Scope</th></tr></thead><tbody id="networkTable"></tbody></table></div></section>
        </div>
        <section id="scenarios" class="section card panel" style="margin-bottom:18px"><div class="panel-head"><div><h3 class="panel-title">Scenario catalog</h3><div class="panel-sub">Registered what-if cases available for governed evaluation</div></div></div><div id="scenarioGrid" class="scenario-grid"></div></section>
        <section class="card panel"><details><summary style="cursor:pointer;color:var(--blue);font-weight:800">Show technical snapshot payloads</summary><pre id="rawPayload" style="white-space:pre-wrap;color:var(--muted);font-size:11px;max-height:420px;overflow:auto;margin-top:15px"></pre></details></section>
        <div class="footer">MINCO · Synthetic Meridian reference case · Human review required · No autonomous operational execution</div>
      </div>
    </main>
  </div>
  <div id="apiKeyPanel" class="api-key"><label for="apiKey">Deployment API key (kept in memory only)</label><input id="apiKey" type="password" placeholder="Optional X-API-Key"><button class="ghost" style="width:100%;margin-top:9px" id="saveApiKey">Apply and refresh</button></div>
  <script>
    const $ = (id) => document.getElementById(id);
    const state = { tower: null, readiness: null, evidence: null, audit: null, feed: null, metrics: null, identity: null, catalog: [], baselineEvaluation: null, counterfactualEvaluation: null, decisionPacket: null, reviewHistory: [], selectedScenario: 'flu_surge' };
    const fmt = (value, digits = 1) => value === null || value === undefined || Number.isNaN(Number(value)) ? '—' : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
    const text = (value, fallback = '—') => value === null || value === undefined || value === '' ? fallback : String(value);
    const esc = (value) => text(value).replace(/[&<>"']/g, (ch) => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#039;' }[ch]));
    const headers = () => { const key = $('apiKey').value.trim(); return key ? { 'X-API-Key': key } : {}; };
    const getJson = async (path) => { const response = await fetch(path, { headers: headers() }); if (!response.ok) throw new Error(`${path} returned HTTP ${response.status}`); return response.json(); };
    const postJson = async (path, payload) => { const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json', ...headers() }, body: JSON.stringify(payload) }); if (!response.ok) { let detail = `HTTP ${response.status}`; try { detail = (await response.json()).detail || detail; } catch (_) {} throw new Error(`${path} returned ${detail}`); } return response.json(); };
    const badge = (id, value, kind = '') => { $(id).textContent = text(value); $(id).className = `badge ${kind}`; };
    const comparisonMetrics = [
      ['Unsafe excess', 'total_unsafe_excess', true, 2],
      ['Overflow excess', 'total_overflow_excess', true, 2],
      ['Max utilization', 'max_utilization_ratio', true, 3],
      ['Blocked arrivals', 'total_blocked_arrivals', true, 2],
      ['Elective rejected', 'total_elective_rejected', true, 2],
      ['Surge activated', 'total_surge_activated', true, 2],
      ['Objective value', 'objective_value', true, 2],
    ];
    function renderWorkbenchCatalog() {
      const options = state.catalog.map(item => `<option value="${esc(item.scenario_id)}">${esc(item.name)}</option>`).join('');
      $('baselineScenario').innerHTML = options;
      $('counterfactualScenario').innerHTML = options;
      $('baselineScenario').value = 'baseline';
      if (!state.catalog.some(item => item.scenario_id === state.selectedScenario)) state.selectedScenario = state.catalog.find(item => item.scenario_id !== 'baseline')?.scenario_id || 'baseline';
      $('counterfactualScenario').value = state.selectedScenario;
    }
    function renderComparison() {
      const baseline = state.baselineEvaluation && state.baselineEvaluation.metrics;
      const counterfactual = state.counterfactualEvaluation && state.counterfactualEvaluation.metrics;
      if (!baseline || !counterfactual) { $('comparisonRows').innerHTML = '<div class="panel-sub" style="padding:16px">No comparison run yet. Choose a counterfactual and run the model.</div>'; return; }
      $('comparisonRows').innerHTML = comparisonMetrics.map(([label, key, lowerBetter, digits]) => { const b = Number(baseline[key] ?? 0); const c = Number(counterfactual[key] ?? 0); const delta = c - b; const good = lowerBetter ? delta <= 0 : delta >= 0; return `<div class="comparison-row"><strong>${esc(label)}</strong><span>${fmt(b, digits)}</span><span>${fmt(c, digits)}</span><span class="${good ? 'delta-good' : 'delta-bad'}">${delta >= 0 ? '+' : ''}${fmt(delta, digits)}</span></div>`; }).join('');
    }
    function renderReviewHistory() {
      const reviews = state.reviewHistory || [];
      $('reviewHistoryRows').innerHTML = reviews.length ? reviews.slice(0, 6).map(item => `<div class="review-event"><strong>${esc(item.decision)} · ${esc(item.reviewed_by)}</strong><span>${esc(item.run_id)} · ${new Date(item.reviewed_at).toLocaleString()}</span></div>`).join('') : '<div class="review-status">No human dispositions recorded in this reference session.</div>';
    }
    function renderPacketStatus() {
      const packet = state.decisionPacket;
      if (!packet) { $('packetStatus').textContent = 'A packet bundles the run manifest, reviews, integrity, and release evidence.'; return; }
      const governance = packet.governance || {};
      const suffix = governance.blockers && governance.blockers.length ? ` · ${governance.blockers.join(', ')}` : '';
      $('packetStatus').textContent = `Packet ${governance.packet_ready ? 'READY' : 'BLOCKED'} · review ${text(governance.review_status, 'PENDING')} · audit ${text(packet.audit_integrity && packet.audit_integrity.status)}${suffix}`;
    }
    async function buildDecisionPacket() {
      const run = state.counterfactualEvaluation;
      if (!run) { $('packetStatus').textContent = 'Run a comparison before building an evidence packet.'; return; }
      const button = $('packetButton'); button.disabled = true; button.textContent = 'Building packet…'; $('packetStatus').textContent = 'Bundling manifest, review history, integrity, and release evidence…';
      try { state.decisionPacket = await getJson(`/v1/decision-packets/${encodeURIComponent(run.run_id)}`); renderPacketStatus(); $('rawPayload').textContent = JSON.stringify({ control_tower: state.tower, readiness: state.readiness, release_evidence: state.evidence, audit: state.audit, event_gateway: state.feed, metrics: state.metrics, baseline_evaluation: state.baselineEvaluation, counterfactual_evaluation: state.counterfactualEvaluation, decision_packet: state.decisionPacket }, null, 2); } catch (error) { state.decisionPacket = null; $('packetStatus').textContent = `Packet failed: ${error.message}`; } finally { button.disabled = false; button.textContent = 'Build evidence packet'; }
    }
    async function runComparison() {
      const button = $('runComparison'); button.disabled = true; button.textContent = 'Running governed comparison…'; $('workbenchStatus').textContent = 'Evaluating baseline and counterfactual. This may take a moment; no operational action is dispatched.';
      try {
        const n = Number($('replications').value || 10); const context = { correlation_id: `workbench-${Date.now()}`, source: 'command_center_workbench' }; const baselineId = $('baselineScenario').value; const counterfactualId = $('counterfactualScenario').value;
        const [baseline, counterfactual] = await Promise.all([postJson('/v1/scenarios/evaluate', { scenario_id: baselineId, n_replications: n, context }), postJson('/v1/scenarios/evaluate', { scenario_id: counterfactualId, n_replications: n, context })]);
        state.baselineEvaluation = baseline; state.counterfactualEvaluation = counterfactual; state.decisionPacket = null; $('reviewRunId').textContent = counterfactual.run_id; $('submitReview').disabled = false; $('packetButton').disabled = false; $('workbenchStatus').textContent = `${counterfactual.scenario.name} evaluated. Review run ${counterfactual.run_id} before recording a disposition.`; renderComparison(); renderPacketStatus();
      } catch (error) { $('workbenchStatus').textContent = `Comparison failed: ${error.message}`; } finally { button.disabled = false; button.textContent = 'Run governed comparison'; }
    }
    async function submitReview() {
      const run = state.counterfactualEvaluation; const reviewer = $('reviewerName').value.trim(); const comment = $('reviewComment').value.trim();
      if (!run) { $('reviewStatus').textContent = 'Run a comparison before recording a review.'; return; }
      if (reviewer.length < 2 || comment.length < 1) { $('reviewStatus').textContent = 'Reviewer identity and rationale are required.'; return; }
      const button = $('submitReview'); button.disabled = true; $('reviewStatus').textContent = 'Writing immutable review event…';
      try { const result = await postJson(`/v1/audit/${encodeURIComponent(run.run_id)}/review`, { reviewed_by: reviewer, decision: $('reviewDecision').value, comment }); state.reviewHistory = await getJson('/v1/audit/reviews?n=10'); state.audit = await getJson('/v1/audit/integrity'); state.decisionPacket = await getJson(`/v1/decision-packets/${encodeURIComponent(run.run_id)}`); $('reviewStatus').textContent = `Recorded ${result.decision}. Event hash ${result.event_hash.slice(0, 16)}…`; renderReviewHistory(); render(); } catch (error) { $('reviewStatus').textContent = `Review failed: ${error.message}`; } finally { button.disabled = false; }
    }
    function render() {
      const t = state.tower, r = state.readiness, e = state.evidence, a = state.audit, f = state.feed;
      $('error').style.display = 'none';
      $('identityPill').textContent = state.identity ? `${text(state.identity.role)} · ${text(state.identity.subject)}` : 'API connected';
      $('snapshotTime').textContent = t ? `Snapshot ${new Date(t.generated_at).toLocaleString()}` : 'Snapshot unavailable';
      $('correlation').textContent = t ? `Correlation ${text(t.correlation_id)}` : '—';
      $('kpiAlert').textContent = text(t && t.overall_alert_level);
      $('kpiGovernance').textContent = text(t && t.governance_state);
      const census = t && t.network && t.network.projected_census ? Object.values(t.network.projected_census).reduce((sum, v) => sum + Number(v || 0), 0) : null;
      $('kpiCensus').textContent = fmt(census, 0);
      $('kpiHospitals').textContent = `${t && t.network && t.network.modeled_hospitals ? t.network.modeled_hospitals.length : 0} modeled hospitals`;
      $('kpiCvar').textContent = fmt(t && t.risk && t.risk.cvar_loss, 2);
      $('kpiEvents').textContent = fmt(t && t.operational_history && t.operational_history.event_count, 0);
      $('kpiFreshness').textContent = text(t && t.operational_history && t.operational_history.freshness_status, 'Unavailable');
      $('towerState').textContent = `${text(t && t.governance_state)} · ${text(t && t.overall_alert_level)} pressure`;
      $('modeledHospitals').textContent = fmt(t && t.diagnostics && t.diagnostics.hospitals_modeled, 0);
      $('selectedActions').textContent = fmt(t && t.decision && t.decision.selected_action_count, 0);
      $('reviewState').textContent = t && t.decision && t.decision.human_review_required ? 'Required' : '—';
      $('claim').textContent = text(t && t.governance && t.governance.claim_boundary);
      const alerts = (t && t.alerts) || [];
      $('alertCount').textContent = `${alerts.length} signal${alerts.length === 1 ? '' : 's'}`;
      $('alerts').innerHTML = alerts.length ? alerts.map(item => `<div class="alert ${String(item.level).toLowerCase()}"><i class="alert-icon"></i><div><strong>${esc(item.code)} · ${esc(item.level)}</strong><span>${esc(item.message)}</span></div></div>`).join('') : '<div class="panel-sub">No active alerts in this snapshot.</div>';
      const actions = (t && t.decision && t.decision.first_stage_actions) || [];
      $('actions').innerHTML = actions.length ? actions.map((item, index) => `<div class="action"><div class="action-num">${index + 1}</div><div><span class="action-name">${esc(item.action || item.action_type || item.type || 'Decision action')}</span><span class="action-detail">${esc(item.hospital_id || item.hospital || item.destination || 'Network scope')}</span></div><div class="action-qty">${esc(item.units ?? item.quantity ?? '')}</div></div>`).join('') : '<div class="panel-sub">No first-stage actions returned.</div>';
      badge('auditBadge', a && a.status, a && a.status === 'VALID' ? '' : 'block'); $('auditText').textContent = a ? `${fmt(a.review_count, 0)} review events · head ${text(a.head_hash, 'GENESIS').slice(0, 16)}` : 'Unavailable';
      badge('evidenceBadge', e && e.status, e && e.status === 'VERIFIED' ? '' : 'block'); $('evidenceText').textContent = e ? `${(e.artifacts || []).length} artifacts · ${text(e.build_fingerprint).slice(0, 16)}…` : 'Unavailable';
      const feedLive = f && f.live_feed_connected; badge('feedBadge', feedLive ? 'CONNECTED' : 'DISCONNECTED', feedLive ? '' : 'warn'); $('feedText').textContent = f ? `${text(f.data_mode)} · ${fmt(f.accepted_event_count, 0)} accepted events` : 'Unavailable';
      const solver = t && t.diagnostics && t.diagnostics.production_solver_preflight; const solverOk = solver && solver.license_verified; badge('solverBadge', solverOk ? 'VERIFIED' : 'PREFLIGHT', solverOk ? '' : 'warn'); $('solverText').textContent = solver ? text(solver.message, 'Dependency posture available') : 'Unavailable';
      $('readinessText').textContent = r ? `${text(r.status)} · ${text(r.blocking_reasons && r.blocking_reasons[0], 'No blocking reason supplied')}` : 'Release posture unavailable.';
      const history = t && t.operational_history; badge('historyBadge', history && history.freshness_status, history && history.freshness_status === 'CURRENT' ? '' : 'warn');
      $('historyRows').innerHTML = history ? [['Data mode', history.data_mode], ['Data stack', history.data_stack], ['Events', fmt(history.event_count, 0)], ['Latest event', history.latest_event_timestamp || '—'], ['Source modes', (history.source_modes || []).join(', ') || '—']].map(row => `<div class="list-row"><strong>${esc(row[0])}</strong><span>${esc(row[1])}</span></div>`).join('') : '<div class="panel-sub">Unavailable</div>';
      const censusMap = t && t.network && t.network.projected_census || {}; $('networkTable').innerHTML = Object.entries(censusMap).map(([hospital, value]) => `<tr><td><strong>${esc(hospital)}</strong></td><td>${fmt(value, 1)}</td><td>Predicted</td></tr>`).join('') || '<tr><td colspan="3">No network census available.</td></tr>';
      const scenarios = (t && t.scenario_catalog) || []; $('scenarioGrid').innerHTML = scenarios.map(item => `<div class="scenario"><strong>${esc(item.name || item.scenario_id)}</strong><p>${esc(item.description)}</p><div class="chips">${Object.entries(item.parameters || {}).slice(0, 4).map(([key, value]) => `<span class="chip">${esc(key)}: ${esc(value)}</span>`).join('')}</div></div>`).join('') || '<div class="panel-sub">No registered scenarios.</div>';
      renderWorkbenchCatalog(); renderComparison(); renderReviewHistory(); renderPacketStatus(); $('rawPayload').textContent = JSON.stringify({ control_tower: t, readiness: r, release_evidence: e, audit: a, event_gateway: f, metrics: state.metrics, identity: state.identity, baseline_evaluation: state.baselineEvaluation, counterfactual_evaluation: state.counterfactualEvaluation, decision_packet: state.decisionPacket }, null, 2);
    }
    async function refresh() {
      $('error').style.display = 'none';
      try {
        const results = await Promise.all([getJson('/v1/control-tower'), getJson('/v1/release-readiness'), getJson('/v1/release-evidence'), getJson('/v1/audit/integrity'), getJson('/v1/operational-events/status'), getJson('/v1/observability/metrics'), getJson('/v1/identity'), getJson('/v1/scenarios'), getJson('/v1/audit/reviews?n=10')]);
        [state.tower, state.readiness, state.evidence, state.audit, state.feed, state.metrics, state.identity, state.catalog, state.reviewHistory] = results; render();
      } catch (error) { $('error').textContent = `Command Center connection issue: ${error.message}. If API-key enforcement is enabled, open API access and provide the deployment key.`; $('error').style.display = 'block'; }
    }
    document.querySelectorAll('.nav button[data-target]').forEach(button => button.addEventListener('click', () => { document.querySelectorAll('.nav button').forEach(item => item.classList.remove('active')); button.classList.add('active'); document.getElementById(button.dataset.target).scrollIntoView({ behavior: 'smooth' }); }));
    $('counterfactualScenario').addEventListener('change', (event) => { state.selectedScenario = event.target.value; }); $('runComparison').addEventListener('click', runComparison); $('submitReview').addEventListener('click', submitReview); $('packetButton').addEventListener('click', buildDecisionPacket); $('refreshButton').addEventListener('click', refresh); $('topRefresh').addEventListener('click', refresh); $('apiKeyButton').addEventListener('click', () => $('apiKeyPanel').classList.toggle('open')); $('saveApiKey').addEventListener('click', () => { $('apiKeyPanel').classList.remove('open'); refresh(); });
    refresh();
  </script>
</body>
</html>"""


def control_tower_html() -> str:
    """Return the static shell; all operational data is loaded from versioned APIs."""
    return CONTROL_TOWER_HTML
