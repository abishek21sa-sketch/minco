MINCO PRODUCT V1
Signature algorithm: FLOW-CVaR

WINDOWS COMMANDS
----------------
.\RUN_ACCEPTANCE.cmd          engineering/math/release gate
.\RUN_PRODUCT_ACCEPTANCE.cmd  new interactive product-runtime gate
.\RUN_DEMO.cmd                deterministic portfolio demo + evidence
.\RUN_APP.cmd                 interactive product surface

Product URL: http://127.0.0.1:8814/

VALIDATION STATE
----------------
Engineering regression in build environment: 109 passed, 3 Gurobi-only skipped in container (previously exercised on Windows)
Product runtime acceptance in build environment: PASS
Windows: PASS on engineering baseline; product-runtime Windows confirmation pending

The engineering baseline is the Windows-passed repository. Product-runtime files were added on top and do not replace the signature algorithm.
Modeled/synthetic/simulated/historical results remain bounded evidence, not realized operational/financial claims.
