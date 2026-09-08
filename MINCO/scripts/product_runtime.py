from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from product_adapter import (  # noqa: E402
    ALGORITHM,
    CONTROLS,
    DEFAULTS,
    DEMO_STRESS,
    PORT,
    PROJECT,
    SUBTITLE,
    THEME,
    compute,
)

ARTIFACT = ROOT / "artifacts" / "product_runtime" / "latest_product_evidence.json"


def prepare_demo() -> dict:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    decision = compute(DEFAULTS)
    counterfactual = compute(DEMO_STRESS)
    payload = {
        "project": PROJECT,
        "algorithm": ALGORITHM,
        "mode": "DETERMINISTIC_PORTFOLIO_DEMO",
        "parameters": DEFAULTS,
        "decision": decision,
        "counterfactual": counterfactual,
        "evidence_boundary": decision["claim"],
        "runtime_contract": "interactive deterministic decision surface over repository-native signature algorithm",
    }
    ARTIFACT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(f"{ALGORITHM}_PRODUCT_DEMO_PREPARED=PASS")
    print(f"DECISION_ID={decision['decision_id']}")
    return payload


def _theme_css() -> str:
    return {
        "aurum": ":root{--bg:#080b12;--panel:#111827;--ink:#e5e7eb;--muted:#9ca3af;--accent:#e3b341;--line:#273246}",
        "circular": ":root{--bg:#f4f7f3;--panel:#ffffff;--ink:#162019;--muted:#66736a;--accent:#237a4b;--line:#d7e2d9}",
        "supply": ":root{--bg:#0c1217;--panel:#13202a;--ink:#e7f0f5;--muted:#94a7b3;--accent:#e37335;--line:#29404f}",
        "minco": ":root{--bg:#f5f7fb;--panel:#ffffff;--ink:#14213d;--muted:#65718a;--accent:#2c6e9b;--line:#d9e0ea}",
        "pdm": ":root{--bg:#0e1012;--panel:#181c20;--ink:#f1f3f5;--muted:#9da7af;--accent:#7fa3b8;--line:#30383f}",
    }[THEME]


def _html() -> str:
    controls = json.dumps(CONTROLS)
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{PROJECT}</title>
<style>{_theme_css()}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Segoe UI,Arial,sans-serif}}
header{{padding:30px 34px 20px;border-bottom:1px solid var(--line)}} .eyebrow{{font-size:12px;letter-spacing:.15em;color:var(--accent);font-weight:700}}
h1{{margin:8px 0 6px;font-size:30px}} .sub{{max-width:1000px;color:var(--muted);line-height:1.5}}
main{{display:grid;grid-template-columns:320px 1fr;gap:18px;padding:20px}} .panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}}
label{{display:block;font-size:13px;margin:14px 0 6px;color:var(--muted)}} input{{width:100%;padding:10px;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:8px}}
button{{width:100%;margin-top:18px;padding:12px;border:0;border-radius:8px;background:var(--accent);color:#111;font-weight:800;cursor:pointer}}
.status{{display:flex;justify-content:space-between;gap:12px;margin-bottom:14px}} .gate{{font-weight:800;color:var(--accent)}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:10px}} .card{{border:1px solid var(--line);padding:14px;border-radius:10px}}
.card small{{color:var(--muted)}} .card strong{{display:block;margin-top:7px;font-size:20px}} h2{{font-size:16px;margin-top:24px}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:left}} th{{color:var(--muted)}}
.claim{{margin-top:18px;padding:12px;border-left:3px solid var(--accent);color:var(--muted);line-height:1.45}} details{{margin-top:16px;color:var(--muted)}} pre{{white-space:pre-wrap;font-size:11px;max-height:340px;overflow:auto}}
@media(max-width:900px){{main{{grid-template-columns:1fr}}.cards{{grid-template-columns:repeat(2,1fr)}}}}
</style></head>
<body><header><div class="eyebrow">{ALGORITHM} · HUMAN-GATED DECISION INTELLIGENCE</div><h1>{PROJECT}</h1><div class="sub">{SUBTITLE}</div></header>
<main><section class="panel"><h2>Decision controls</h2><div id="controls"></div><button onclick="runDecision()">Recompute decision</button><p style="font-size:12px;color:var(--muted)">This surface calls the repository-native signature algorithm. Results are model outputs, not autonomous actions.</p></section>
<section class="panel"><div class="status"><span id="decisionId">—</span><span class="gate" id="gate">—</span></div><div class="cards" id="metrics"></div><h2>Recommended decision objects</h2><div id="actions"></div><h2>Baselines / evidence context</h2><div id="baselines"></div><div class="claim" id="claim"></div><details><summary>Raw auditable payload</summary><pre id="raw"></pre></details></section></main>
<script>
const controls={controls};
function esc(x){{return String(x??'').replace(/[&<>]/g,s=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[s]))}}
function mkTable(rows){{if(!rows||!rows.length)return '<em>No discrete actions in this decision.</em>';const keys=Object.keys(rows[0]);return '<table><thead><tr>'+keys.map(k=>'<th>'+esc(k)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+keys.map(k=>'<td>'+esc(r[k])+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}}
document.getElementById('controls').innerHTML=controls.map(c=>`<label>${{c.label}}</label><input id="${{c.key}}" type="number" min="${{c.min}}" max="${{c.max}}" step="${{c.step}}" value="${{c.default}}">`).join('');
async function runDecision(){{const q=new URLSearchParams();controls.forEach(c=>q.set(c.key,document.getElementById(c.key).value));const d=await (await fetch('/api/decision?'+q.toString())).json();render(d)}}
function render(d){{document.getElementById('decisionId').textContent=d.decision_id;document.getElementById('gate').textContent=d.gate;document.getElementById('metrics').innerHTML=d.metrics.map(x=>`<div class="card"><small>${{esc(x[0])}}</small><strong>${{esc(x[1])}}</strong></div>`).join('');document.getElementById('actions').innerHTML=mkTable(d.actions);document.getElementById('baselines').innerHTML='<table><tbody>'+d.baselines.map(x=>`<tr><th>${{esc(x[0])}}</th><td>${{esc(x[1])}}</td></tr>`).join('')+'</tbody></table>';document.getElementById('claim').textContent=d.claim;document.getElementById('raw').textContent=JSON.stringify(d.raw,null,2)}}
fetch('/api/evidence').then(r=>r.json()).then(x=>render(x.decision));
</script></body></html>"""
    bridge = """<script>
window.__MINCO_API_BASE__=(window.__MINCO_API_BASE__||((location.hostname==='localhost'||location.hostname==='127.0.0.1')?'':'https://minco-healthcare-api.onrender.com')).replace(/\/$/,'');
const _mincoFetch=window.fetch.bind(window);
window.fetch=(input,init)=>{const url=typeof input==='string'?input:input.url;return url.startsWith('/api/')?_mincoFetch(window.__MINCO_API_BASE__+url,init):_mincoFetch(input,init)};
</script>"""
    return page.replace("</head>", bridge + "</head>", 1)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", os.getenv("MINCO_CORS_ORIGIN", "*"))
        self.send_header("Access-Control-Allow-Methods", "GET,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,X-Request-ID")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send(204, b"")

    def do_GET(self) -> None:
        p = urlparse(self.path)
        try:
            if p.path == "/health":
                return self._send(
                    200,
                    json.dumps(
                        {"status": "ok", "project": PROJECT, "algorithm": ALGORITHM}
                    ).encode(),
                )
            if p.path == "/":
                return self._send(200, _html().encode("utf-8"), "text/html; charset=utf-8")
            if p.path == "/api/evidence":
                payload = (
                    json.loads(ARTIFACT.read_text(encoding="utf-8"))
                    if ARTIFACT.exists()
                    else prepare_demo()
                )
                return self._send(200, json.dumps(payload, default=str).encode())
            if p.path == "/api/decision":
                q = parse_qs(p.query)
                params = {c["key"]: float(q.get(c["key"], [c["default"]])[0]) for c in CONTROLS}
                d = compute(params)
                ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
                ARTIFACT.write_text(
                    json.dumps(
                        {
                            "project": PROJECT,
                            "algorithm": ALGORITHM,
                            "mode": "INTERACTIVE",
                            "parameters": params,
                            "decision": d,
                        },
                        indent=2,
                        sort_keys=True,
                        default=str,
                    ),
                    encoding="utf-8",
                )
                return self._send(200, json.dumps(d, default=str).encode())
            if p.path == "/download/evidence.json":
                payload = (
                    ARTIFACT.read_bytes()
                    if ARTIFACT.exists()
                    else json.dumps(prepare_demo(), indent=2).encode()
                )
                return self._send(200, payload, "application/json")
            return self._send(404, b'{"detail":"not found"}')
        except Exception as exc:
            return self._send(
                500, json.dumps({"error": type(exc).__name__, "detail": str(exc)}).encode()
            )

    def log_message(self, fmt: str, *args) -> None:
        print("[product]", fmt % args)


def serve(*, open_browser: bool = True) -> None:
    if not ARTIFACT.exists():
        prepare_demo()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", str(PORT)))
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"PRODUCT_RUNTIME_READY={url}", flush=True)
    if open_browser:
        webbrowser.open(url, new=2)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping product runtime...")
    finally:
        server.server_close()


def acceptance() -> None:
    payload = prepare_demo()
    if payload["decision"]["gate"] not in ("AUTHORIZED", "RESEARCH_ONLY"):
        raise SystemExit(f"PRODUCT_RUNTIME_ACCEPTANCE=FAIL gate={payload['decision']['gate']}")
    changed = compute(DEMO_STRESS)
    if (
        changed["decision_id"] == payload["decision"]["decision_id"]
        and changed["raw"] == payload["decision"]["raw"]
    ):
        raise SystemExit("PRODUCT_RUNTIME_ACCEPTANCE=FAIL counterfactual did not change")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for path in ("/health", "/", "/api/evidence"):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=20) as response:
                if response.status != 200:
                    raise SystemExit(
                        f"PRODUCT_RUNTIME_ACCEPTANCE=FAIL http={path}:{response.status}"
                    )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    if not ARTIFACT.exists():
        raise SystemExit("PRODUCT_RUNTIME_ACCEPTANCE=FAIL evidence artifact missing")
    print(f"{ALGORITHM}_PRODUCT_RUNTIME_ACCEPTANCE=PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-demo", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--accept", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if args.prepare_demo:
        prepare_demo()
    if args.accept:
        acceptance()
    if args.serve:
        serve(open_browser=not args.no_browser)
    if not (args.prepare_demo or args.accept or args.serve):
        ap.print_help()


if __name__ == "__main__":
    main()
