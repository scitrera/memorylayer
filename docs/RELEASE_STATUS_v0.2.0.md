# v0.2.0 publication status

Release commit: `6ee99813f287ff7c20fe85f427032558f1bd56d4`.

The root tag and both coordinated Go module tags are published at this commit.
Both Go modules were downloaded through the public module proxy and compiled
in an independent consumer project.

The server, CPU embed server, and CUDA 13 embed server images are published to
GHCR and Docker Hub. All six native AMD64/ARM64 build jobs and all three manifest
merge jobs passed. Each release manifest contains both Linux architectures;
manifests match across registries. An offline server-container smoke check
verified health, 13 RPG API paths, and Aether client 0.2.3. Live GPU inference
and external-service end-to-end tests were not performed.

The PyPI trusted-publisher update worked: all five uploads completed. Downloaded
`memorylayer-server==0.2.0` and `memorylayer-server-rpg==0.2.0` wheels contain
their expected Python source and match the released checkout.

Downloaded 0.2.0 wheels for `memorylayer-client`, `memorylayer-langchain`, and
`memorylayer-llamaindex` contain metadata but no Python modules. These releases
are now yanked. The approved 0.2.1 correction is published for those
three packages. Downloaded wheels match the reviewed source, have correct
dependency pins, and import successfully after installation. See
[the Python patch review](RELEASE_PATCH_REVIEW_python_v0.2.1.md).

All four npm v0.2.0 packages are published and verified. The owner corrected
trusted-publisher settings after the workflow moved from `release.yml` to
`publish-npm.yml` with the `npm` environment. Retrying the existing release
workflow succeeded. All four registry records identify GitHub OIDC publishers
and include provenance. Tarball hashes, package contents, entry points, licenses,
and registry dependency references passed verification. All four packages import
successfully after a fresh registry install. npm verified all 98 installed
packages' registry signatures and 13 attestations.

Repo-tools' existing `ci.npm.use_oidc: true` setting is enabled on main, removing
legacy token injection from future jobs; all hosted checks passed. Correcting
this configuration did not require another repo-tools release.

The [GitHub v0.2.0 release](https://github.com/scitrera/memorylayer/releases/tag/v0.2.0)
is public. Existing release tags have not been replaced.

Publication workflows:

- [Python 0.2.1 correction: completed and verified](https://github.com/scitrera/memorylayer/actions/runs/34310308630)
- [Python 0.2.0 server/RPG: uploads completed](https://github.com/scitrera/memorylayer/actions/runs/34307626555)
- [Go: completed](https://github.com/scitrera/memorylayer/actions/runs/34307626533)
- [Native multiarch containers: completed](https://github.com/scitrera/memorylayer/actions/runs/34307626355)
- [npm: completed and verified](https://github.com/scitrera/memorylayer/actions/runs/34307626537)
