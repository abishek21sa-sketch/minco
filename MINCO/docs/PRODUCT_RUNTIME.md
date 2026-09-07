# Product Runtime

MINCO — Hospital Capacity Operations Workstation — Product Runtime RC1

COMMANDS
--------
RUN_ACCEPTANCE.cmd          Existing Windows engineering/release acceptance gate.
RUN_APP.cmd                 Launch the interactive FLOW-CVaR product surface.
RUN_DEMO.cmd                Regenerate deterministic demo evidence, then launch the product.
RUN_PRODUCT_ACCEPTANCE.cmd  Verify algorithm invocation, parameter counterfactuals, evidence artifact, and HTTP lifecycle.

PRODUCT SURFACE
---------------
URL: http://127.0.0.1:8814/
Evidence artifact: MINCO/artifacts/product_runtime/latest_product_evidence.json
Stop the app with Ctrl+C in its command window.

CLAIM BOUNDARY
--------------
The runtime preserves each repository's human-review gate and source claim boundary. Modeled, synthetic, simulated, historical, optimized, and realized evidence are not conflated.
