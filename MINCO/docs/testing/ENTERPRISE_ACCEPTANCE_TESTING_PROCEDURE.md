# MINCO Enterprise Candidate Testing Procedure

This procedure validates the engineering package and governance controls. It does not approve MINCO for clinical, live-hospital, autonomous, or production use. Current evidence is for the synthetic Meridian reference case.

## 1. Automated acceptance

Open PowerShell and run:

```powershell
Set-Location 'C:\Users\babbi\OneDrive\Desktop\MINCO_PRODUCT_V1'
cmd /c RUN_ENTERPRISE_ACCEPTANCE.cmd
```

Expected result:

```text
{"status": "PASS", "report_path": "...\\MINCO\\results\\validation\\enterprise_acceptance_report.json"}
```

The harness runs:

1. Product runtime acceptance.
2. The complete Python regression suite.
3. The 459-event DuckDB/Parquet/Polars Phase 2 backend gate.
4. Public contract-registry regeneration and the 19-schema check.
5. Persisted release-evidence generation.
6. In-process API governance smoke checks.

Open the resulting report at:

```text
C:\Users\babbi\OneDrive\Desktop\MINCO_PRODUCT_V1\MINCO\results\validation\enterprise_acceptance_report.json
```

Do not treat a `PASS` here as production approval. It means the candidate package is internally consistent and its declared controls behave as tested.

## 2. Live API smoke test

Keep PowerShell window 1 open and start the product:

```powershell
Set-Location 'C:\Users\babbi\OneDrive\Desktop\MINCO_PRODUCT_V1'
cmd /c RUN_APP.cmd
```

In PowerShell window 2, run:

```powershell
$base = 'http://127.0.0.1:8814'
$headers = @{ 'X-Correlation-ID' = 'manual-enterprise-01' }

Invoke-RestMethod "$base/health" -Headers $headers
Invoke-RestMethod "$base/readiness" -Headers $headers
Invoke-RestMethod "$base/v1/release-evidence" -Headers $headers
Invoke-RestMethod "$base/v1/release-readiness" -Headers $headers
Invoke-RestMethod "$base/v1/audit/integrity" -Headers $headers
Invoke-RestMethod "$base/v1/operational-events/status" -Headers $headers
(Invoke-WebRequest "$base/metrics").Content
```

Expected outcomes:

- `/health`: HTTP 200 and the same `X-Correlation-ID`.
- `/readiness`: HTTP 200; `status` may be `degraded` when the licensed solver is not verified.
- `/v1/release-evidence`: HTTP 200, `status = VERIFIED`, no missing artifacts, and a 64-character `build_fingerprint`.
- `/v1/release-readiness`: HTTP 200, `status = BLOCKED`, `release_evidence_attestation_verified = true`, and `autonomous_execution_permitted = false`.
- `/v1/audit/integrity`: HTTP 200 and `status = VALID`.
- `/v1/operational-events/status`: HTTP 200; the untouched reference package reports `live_feed_connected = false`.
- `/metrics`: text containing `minco_api_requests_total`.

## 3. Operations Workbench comparison and review

Open `http://127.0.0.1:8814/` and select **Operations Workbench**. Choose `baseline` and a counterfactual such as `flu_surge`, leave replications at `10`, and select **Run governed comparison**.

Expected outcomes:

- both scenario evaluations complete and display a comparison table with baseline, counterfactual, and delta columns;
- the counterfactual receives a governed run identifier and becomes selectable in the human-review panel;
- entering a reviewer name and rationale, then selecting `DEFERRED`, `REJECTED`, or `ACCEPTED_FOR_OPERATIONS_REVIEW`, records the disposition without authorizing execution;
- the review history refreshes and `/v1/audit/integrity` remains `VALID`.

For an API-level check, use:

```powershell
$scenarios = Invoke-RestMethod "$base/v1/scenarios" -Headers $headers
$evalBody = @{ scenario_id = 'flu_surge'; n_replications = 10; context = @{ correlation_id = 'manual-workbench-01'; source = 'manual-workbench' } } | ConvertTo-Json -Depth 10
$evaluation = Invoke-RestMethod "$base/v1/scenarios/evaluate" -Method Post -Headers $headers -ContentType 'application/json' -Body $evalBody
$reviewBody = @{ reviewed_by = 'operations-reviewer'; decision = 'DEFERRED'; comment = 'Manual Workbench governance check.' } | ConvertTo-Json
Invoke-RestMethod "$base/v1/audit/$($evaluation.run_id)/review" -Method Post -Headers $headers -ContentType 'application/json' -Body $reviewBody
Invoke-RestMethod "$base/v1/audit/integrity" -Headers $headers
```

The scenario comparison is an auditable decision-support workflow. It does not create staffing, transfer, diversion, clinical, or other autonomous execution commands.

After recording the review, build the portable evidence packet:

```powershell
$packet = Invoke-RestMethod "$base/v1/decision-packets/$($evaluation.run_id)" -Headers $headers
$packet.governance
$packet.audit_integrity
$packet.release_evidence.status
```

Expected: `packet_ready = true`, `review_status = DEFERRED`, audit status `VALID`, release evidence `VERIFIED`, and `autonomous_execution_permitted = false`. The packet is a review artifact, not an execution command.

## 4. API-key boundary test

Stop the app with `Ctrl+C` in window 1. Start it with a temporary local key:

```powershell
$env:MINCO_API_KEY = 'local-test-key-change-me'
Set-Location 'C:\Users\babbi\OneDrive\Desktop\MINCO_PRODUCT_V1'
cmd /c RUN_APP.cmd
```

In window 2, verify that liveness remains public while operational routes require the key:

```powershell
$base = 'http://127.0.0.1:8814'
Invoke-RestMethod "$base/health"

try {
    Invoke-WebRequest "$base/v1/release-evidence" -UseBasicParsing
} catch {
    $_.Exception.Response.StatusCode.value__
}

$auth = @{ 'X-API-Key' = 'local-test-key-change-me' }
Invoke-RestMethod "$base/v1/release-evidence" -Headers $auth
```

Expected results: `/health` returns 200, the unauthenticated release-evidence request returns 401, and the authenticated request returns `status = VERIFIED`. Clear the temporary key after stopping the app:

```powershell
Remove-Item Env:MINCO_API_KEY -ErrorAction SilentlyContinue
```

### Role-based authorization test

For a local deployment-boundary test, use a temporary multi-role map. The values below are synthetic test credentials only:

```powershell
$env:MINCO_API_KEYS = '{"viewer-local":{"subject":"command-center-viewer","role":"viewer"},"analyst-local":{"subject":"capacity-analyst","role":"analyst"},"reviewer-local":{"subject":"operations-reviewer","role":"reviewer"}}'
Set-Location 'C:\Users\babbi\OneDrive\Desktop\MINCO_PRODUCT_V1'
cmd /c RUN_APP.cmd
```

Verify identity and least privilege:

```powershell
$base = 'http://127.0.0.1:8814'
$viewer = @{ 'X-API-Key' = 'viewer-local' }
$analyst = @{ 'Authorization' = 'Bearer analyst-local' }
$reviewer = @{ 'X-API-Key' = 'reviewer-local' }

Invoke-RestMethod "$base/v1/identity" -Headers $viewer
Invoke-RestMethod "$base/v1/scenarios" -Headers $viewer

try {
    Invoke-RestMethod "$base/v1/scenarios/evaluate" -Method Post -Headers $viewer -ContentType 'application/json' -Body '{"scenario_id":"flu_surge","n_replications":2}'
} catch {
    $_.Exception.Response.StatusCode.value__
}
```

Expected: `/v1/identity` reports `role = viewer`; scenario catalog access returns 200; scenario evaluation with the viewer credential returns 403 with `code = insufficient_role`. Analyst credentials may evaluate scenarios. Only reviewer/admin credentials may record a human disposition. Clear the map after testing:

```powershell
Remove-Item Env:MINCO_API_KEYS -ErrorAction SilentlyContinue
```

## 5. Governed ingestion test

With the app running, submit one synthetic de-identified external-stream event:

```powershell
$base = 'http://127.0.0.1:8814'
$headers = @{ 'X-Correlation-ID' = 'manual-ingest-01' }
$now = (Get-Date).ToUniversalTime().ToString('o')
$validEvent = @{
    event_id = 'manual-enterprise-event-001'
    event_timestamp = $now
    ingested_at = $now
    hospital_id = 'meridian-central'
    event_type = 'arrival'
    patient_pathway_or_cohort = 'adult_medical'
    resource = 'ed'
    quantity = 1
    source_system = 'manual-test'
    source_mode = 'external_stream'
}
$body = @{
    source_system = 'manual-test'
    source_mode = 'external_stream'
    received_at = $now
    events = @($validEvent)
} | ConvertTo-Json -Depth 10

Invoke-RestMethod "$base/v1/operational-events/ingest" -Method Post -Headers $headers -ContentType 'application/json' -Body $body
Invoke-RestMethod "$base/v1/operational-events/status" -Headers $headers
```

Expected: HTTP 200, `status = ACCEPTED`, `accepted_count = 1`, and the subsequent status shows `live_feed_connected = true` for that running process. This is a synthetic boundary test, not a real ADT/EHR connection.

Now verify raw patient identifiers are rejected:

```powershell
$invalidEvent = $validEvent.Clone()
$invalidEvent.event_id = 'manual-enterprise-event-002'
$invalidEvent.patient_id = 'TEST-DO-NOT-USE-IN-PRODUCTION'
$invalidBody = @{
    source_system = 'manual-test'
    source_mode = 'external_stream'
    received_at = $now
    events = @($invalidEvent)
} | ConvertTo-Json -Depth 10

Invoke-RestMethod "$base/v1/operational-events/ingest" -Method Post -Headers $headers -ContentType 'application/json' -Body $invalidBody
```

Expected: HTTP 200 with `status = REJECTED`, `accepted_count = 0`, `rejected_count = 1`, and a rejection detail stating that `patient_id` is not accepted. Restart the app afterward to return the in-memory reference gateway to its disconnected baseline.

## 6. Evidence review checklist

Before sharing the package, confirm:

- `enterprise_acceptance_report.json` has `status = PASS`.
- `release_evidence_attestation.json` has `status = VERIFIED` and `missing_artifacts = []`.
- `PRODUCT_V1_RELEASE_MANIFEST.json` reports 135 passed tests and 20 contracts.
- The release-readiness scorecard remains `BLOCKED` until live ADT/EHR, enterprise IAM, production solver, external validation, and formal governance gates are completed.
- No real patient identifiers or production credentials are used in testing.
