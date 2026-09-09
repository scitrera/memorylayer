# v0.2.0 release review — 2026-09-08

Status: the owner approved pushing and tagging MemoryLayer v0.2.0 on 2026-09-08,
after reviewing the disclosed NLTK advisory and validation limits below.
This document records the release review; GitHub Actions and the release page
report publication status. Repo-tools v0.1.30 is published, and its wheel passes
MemoryLayer's version and generated-workflow checks.

## Release source and coordinated artifacts

The reviewed source includes 217 local commits beyond the public branch plus
pending work on deterministic memory, typed relations, session checkpoints,
context packs, ingestion, RPG, embedding providers, and SDKs/plugins.

**Publish from the sanitized release snapshot, not the existing develop
history.** An earlier unpublished commit contains a private network hostname,
even though it was subsequently removed. A clean snapshot was prepared on top of
the verified public main/develop commit
`de0cbf7def0e4c73175660847e2c50fe9600c3ee`, preserving public ancestry without
including those unpublished intermediate commits. The original working branch
and its existing changes are preserved.

The following tags must identify the same approved release commit:

- `v0.2.0`
- `memorylayer-sdk-go/v0.2.0`
- `memorylayer-sdk-go/aether/v0.2.0`

The nested Aether module remains separate. It now uses Aether API/SDK v0.2.3,
and both Go modules now require Go 1.25.14, matching CI. The core module still
has no external dependencies. Repo-tools already handles the coordinated tags:
`ci.go.module_tags: push` generates a workflow that tests the Go modules,
verifies the release version, and creates both module tags from the root tag.
No manual sibling tagging is needed. The upstream fix additionally refuses
branch dispatches and publishes missing tags atomically, after checking all
existing tags for conflicts.

Publication scope is now five PyPI packages and four npm packages, plus the
configured container images and the two Go modules. `memorylayer-server-rpg`
joins the v0.2.0 PyPI release and is published after its exact-version dependency,
`memorylayer-server==0.2.0`. RPG remains a separate plugin and is also included
in the main server container. OpenClaw is tested and included in source but is
not published to npm. Explorer remains an unpublished application.

### First RPG publication setup

The public PyPI JSON API returned 404 for `memorylayer-server-rpg` during this
review. The owner confirmed that the pending trusted publisher is configured
with these values:

| Setting | Value |
| --- | --- |
| PyPI project | `memorylayer-server-rpg` |
| GitHub owner | `scitrera` |
| Repository | `memorylayer` |
| Workflow filename | `publish-python.yml` |
| GitHub environment | `pypi` |

The workflow uses PyPI trusted publishing. Pending-publisher registration was
confirmed by the owner; this session did not change the PyPI account settings.
It allows the first approved upload to create the project; it does not publish
the package by itself. See [PyPI's pending-publisher instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Issues corrected

- **Distribution data leaks:** actual npm tarball previews included local agent
  sessions/replays, and a built Python SDK wheel included agent state under its
  source package. Explicit npm file lists and Python build exclusions remove
  these, credentials, local databases, and caches. Git/Docker exclusions were
  strengthened, and distributable packages now include the license text.
- **Private references:** removed an unpublished benchmark's private VPN URL, a
  developer-specific benchmark default, and an Explorer development-machine
  hostname. Benchmark relations now require an explicit corpus path.
- **Broken Python adapters:** declared `langchain-classic`, handled the SDK's
  public exceptions, and corrected obsolete response fixtures. Both adapters
  now participate in release CI against the local SDK. A Python SDK test now
  owns its event loop instead of relying on pytest's ambient loop.
- **Authorization responses:** all 14 RPG routes preserve HTTP authorization
  denials instead of returning 500; regression tests also assert denied calls
  never reach graph services.
- **Browser SDK failure:** filesystem-only skill imports prevented Explorer's
  production build. Browser dependency mappings and explicit runtime checks
  preserve browser use of the client and give clear errors for Node-only helpers.
- **Dependency security:** refreshed compatible npm dependencies, overrode the
  affected transitive gRPC/PostCSS versions, and upgraded `python-multipart` to
  0.0.31. OpenClaw uses its current public memory-capability types and declares
  OpenClaw >=2026.9.3 / Node >=24.16.0. Lockfile versions are synchronized.
- **Broken npm publish builds:** release-mode version rewriting retains local
  links in lockfiles. The generated publish jobs lacked sibling `dist` artifacts
  and failed to compile. The general fix is released in repo-tools v0.1.30:
  synchronize versions while retaining local references, build each declared
  sibling dependency in dependency order, install the publishing package, then
  rewrite references to registry pins before its final build and publication.
  This works with or without lockfiles and does not depend on test artifacts or
  upstream package propagation. Inlined npm test gates also retain their
  dependency graph. `ci.npm.require_matching_tag: true` rejects branches and
  mismatched tags. All eight workflows, including npm publication, are generated
  again; MemoryLayer pins the published repo-tools v0.1.30.

## Validation evidence

| Check | Result |
| --- | --- |
| Original-environment core suite | 2,125 passed; 14 skipped; 331 deselected |
| Clean Python 3.12 wheel core suite | 2,049 passed; 21 skipped; 331 deselected; optional-provider availability changes collection |
| Clean wheel suites | SDK 105; LangChain 77; LlamaIndex 67; RPG 91; embed server 140 passed |
| Changed API integration contracts | 9 passed, including checkpoint/delta, knowledge-work/email ingest, and execution views |
| Retrieval gate | PASS; 10/10 queries scored, no errors, recall@5 and MRR 1.000 |
| Patched multipart parser | FastAPI file upload and typed form round trip passed |
| Node 24.16.0 | All five library/plugin builds passed; SDK 120, MCP 60, OpenClaw 34 tests passed |
| Explorer | Type check and Next.js production build passed; all 10 pages generated |
| npm publication rehearsal | All four publishable packages build and pack from fresh checkouts using the new generated steps, with no prebuilt siblings; tarballs have registry dependency pins |
| Package inspection | Six wheels/sdists built; five npm tarballs checked for entry points, license text, local-state exclusion, and registry dependency pins |
| RPG first-release rehearsal | Fresh wheel/sdist pass strict Twine checks; an isolated sdist rebuild matches every wheel member. Installed-wheel suite: 91 passed. Automatic discovery registers 13 API paths and the RPG ontology; publication waits for core, tests, and tag validation |
| Go | Both modules pass vet and tests with race detection on Go 1.25.14; Aether transport uses v0.2.3 |
| Repo-tools regression tests | 549 passed; Ruff and version/workflow checks passed. Fresh-checkout npm rehearsals fail against 0.1.29 and pass with the fix, both with and without lockfiles |
| Repo-tools publication | Hosted release checks passed on Python 3.11, 3.12, and 3.13; PyPI wheel checksum verified, all 45 Python source files match the release, and the published wheel passes MemoryLayer's version and all eight workflow checks |
| Go tag publication rehearsal | Disposable local remotes verify same-commit tags, retries, existing conflicts, atomic rejection, and branch rejection |
| Go vulnerability analysis | No reachable vulnerability reported for the Aether transport; advisory matches remain in unused dependency APIs |
| npm audit | Zero reported vulnerabilities across all six npm projects, including development dependencies |
| Python OSV scan | 156 installed dependencies scanned; only the NLTK advisory below remains |
| Metadata/workflows | Version check, generated-workflow drift check, publish shell syntax, and matching/mismatched-tag guards passed |
| Source privacy | Common credential/JWT/key patterns and private-host/developer-path scan found no remaining matches in intended release source |

The package and clean-install checks were performed in temporary directories,
without replacing the developer's installed dependency trees. The exact pinned
release-checker source was run locally inside the sandbox, then the same checks
were repeated with the checksum-verified repo-tools v0.1.30 wheel from PyPI.
The upstream prerequisite is complete:
[repo-tools v0.1.30](https://github.com/scitrera/repo-tools/releases/tag/v0.1.30).

## Remaining risks and limits

- **Unfixed upstream NLTK advisory:** LlamaIndex brings in NLTK 3.10.3, which is
  still the newest published version. OSV reports model-artifact APIs bypassing
  allowed filesystem roots; no fixed version is listed. This adapter does not
  directly invoke those APIs, but that does not establish safety for applications
  using other NLTK/LlamaIndex features. Treat it as a disclosed dependency risk,
  avoid untrusted model artifacts, and update when a fix is published.
  [Advisory GHSA-8mgp-746c-j5xp](https://github.com/advisories/GHSA-8mgp-746c-j5xp).
- GPU/live-provider tests, real OpenClaw/Aether service end-to-end tests, and
  multi-architecture Docker builds were not run. Mock/offline suites and source
  packaging do not establish those deployment properties.
- Existing Python style/import lint debt and deprecation/mock warnings remain;
  Python lint is still disabled in generated CI. Changed adapter/authorization
  files pass their targeted Ruff check.
- The privacy scan covers intended source, package contents, and incoming commit
  patches; it is not a guarantee against every possible secret format or a
  forensic audit of already-public history.
- The legacy `scripts/rollover-release.sh` resets/rebases branches and can replace
  tags. It is not the proposed path for this release; use the reviewed snapshot
  and create new tags only after approval.

## Approval boundary

The owner explicitly approved pushing and tagging v0.2.0 and confirmed the RPG
pending trusted publisher. Publish from the reviewed sanitized snapshot and
require hosted CI to pass before tagging. The generated Go workflow creates
both module tags at that same commit. Never replace an existing release tag.
