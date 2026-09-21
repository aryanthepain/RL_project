## Summary of Changes

<!-- Provide a concise description of the motivation, changes made, and impact. -->

## Linked Issues

<!-- 
Mandatory convention: All PRs must link to an existing GitHub issue.
Examples: Closes #12, Fixes #13
-->
Closes #

## Compute & Verification Checklist

- [ ] Deterministic unit test suite passes: `python -m pytest` (105/105 passing)
- [ ] No large binary artifacts committed (`.pt`, `.zip`, `.mp4`, or checkpoint files)
- [ ] Code follows PEP 8 standards with type annotations and docstrings
- [ ] Any new CLI flags or configuration parameters documented in `README.md`
- [ ] If training was executed, run metadata and metrics verified via `scripts/clean_runs.py --inspect`

## Reviewer Notes

<!-- Mention specific areas where peer review or scrutiny is requested. -->
