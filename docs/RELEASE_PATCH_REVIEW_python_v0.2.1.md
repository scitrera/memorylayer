# Python packaging correction: 0.2.1

Status: published and verified after owner approval. All hosted checks and
all Python publication jobs passed.

The uploaded 0.2.0 wheels for `memorylayer-client`, `memorylayer-langchain`,
and `memorylayer-llamaindex` contain metadata but no Python modules. Their
Hatch sdist configuration used `packages`, which stripped the `src/` prefix.
CI then built wheels from those sdists using paths that no longer existed.
The earlier direct wheel checks did not exercise this failing build path.

The correction releases those three Python packages as 0.2.1 and
updates both adapters to require `memorylayer-client==0.2.1`. The server and
RPG packages remain at 0.2.0. Existing Go, npm, container versions and release
tags are preserved. PyPI does not permit replacement of uploaded filenames;
the three defective 0.2.0 Python releases have been yanked.

The fix uses sdist `include` paths to preserve `src/`. A generated CI step now
runs `scripts/check-python-package.py` for every published Python project.
It invokes the same sdist-to-wheel build as publication, verifies original
source paths in the sdist, compares every wheel Python file with the source,
and checks that the license is included.

Validation completed:

- Corrected SDK and adapter wheels contain 15, 3, and 2 Python files,
  respectively, matching their source checkouts byte for byte.
- Tests against installed replacement wheels, outside their source directories:
  SDK 105 passed, LangChain 77 passed, LlamaIndex 67 passed.
- Strict Twine validation passes for all three wheels and all three sdists.
- The new check rejects the original v0.2.0 SDK configuration and passes the
  corrected packages as well as the server and RPG packages.
- Version synchronization and generated-workflow checks pass.

Publication used the reviewed commit
`9e7ba462de4d1eda3fdcb222c387749bde7fe1dd` through the
[Python publication workflow](https://github.com/scitrera/memorylayer/actions/runs/34310308630).
Downloaded PyPI wheels passed checksum, exact source-content, license, and
dependency-pin verification; all three import successfully after installation.
No existing tag was replaced. The PyPI JSON API confirms that all three
defective 0.2.0 releases are now yanked.
