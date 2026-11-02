# Release and open-source readiness checklist

- [x] Project owner confirmed rights to the original source code and authorized MIT licensing (November 2026).
- [x] Publish the project-level OSI-approved MIT LICENSE file for the original code.
- [ ] Verify rights to bundled PDFs and the Streamline/TradingView SVG separately; see [third-party notices](../THIRD_PARTY_NOTICES.md). Do not treat these as MIT-licensed by default.
- [ ] Confirm redistributability of any additional project documentation or assets before packaging a release.
- [ ] Check Git history, commits, and documents for credentials, personal data, confidential material, and unlicensed content.
- [ ] Test setup on a clean Linux or Windows installation.
- [ ] Run `python -m unittest discover -s tests -v` and verify CI passes.
- [ ] Validate opening-range timestamps and market sessions with realistic and DST fixtures.
- [ ] Document data-provider availability limits, missing data policy, and known assumptions.
- [ ] Publish sample output using synthetic or redistribution-authorized data.
- [ ] Create release notes and a tagged version only after these checks.

Open-source program applications should use real usage evidence, not synthetic activity or unverified adoption counts.
