# Release and open-source readiness checklist

- [ ] Confirm copyright ownership and permission to publish all code and included documents.
- [ ] Add an OSI-approved project-level LICENSE file after confirming rights.
- [ ] Check Git history, commits, and documents for credentials, personal data, confidential material, and unlicensed content.
- [ ] Test setup on a clean Linux or Windows installation.
- [ ] Run `python -m unittest discover -s tests -v` and verify CI passes.
- [ ] Validate opening-range timestamps and market sessions with realistic and DST fixtures.
- [ ] Document data-provider availability limits, missing data policy, and known assumptions.
- [ ] Publish sample output using synthetic or redistribution-authorized data.
- [ ] Create release notes and a tagged version only after these checks.

Open-source program applications should use real usage evidence, not synthetic activity or unverified adoption counts.
