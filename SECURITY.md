# Security Policy

The current MINCO release is a synthetic-data research platform and is not approved for patient-level data or production hospital deployment.

Do not submit protected health information, credentials, private keys, proprietary hospital data, or licensed solver files to the repository.

The candidate includes an optional deployment boundary: when `MINCO_API_KEY` is configured, operational routes require `X-API-Key` and rejected requests retain correlation metadata for traceability. This is a baseline control only. Production use still requires enterprise identity and role-based authorization, encrypted transport, managed secret rotation, audit identity, privacy review, network policy, and deployment-specific governance that are not present in this release.

`/health`, `/readiness`, documentation routes, and CORS preflight remain available without the key so platform probes can operate. An unset key deliberately leaves the local reference package in `LOCAL_REFERENCE_CASE_NO_AUTH` mode; the release-readiness scorecard reports that mode as a production blocker.
