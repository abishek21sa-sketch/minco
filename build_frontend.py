import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "MINCO"
sys.path.insert(0, str(PROJECT / "scripts"))
from product_runtime import _html  # noqa: E402

(ROOT / "site").mkdir(parents=True, exist_ok=True)
(ROOT / "site" / "index.html").write_text(_html(), encoding="utf-8")
print("MINCO_FRONTEND_BUILD=PASS")
