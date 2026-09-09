#!/usr/bin/env python3
"""Check the sdist-to-wheel build used by publication, including package code."""

import argparse
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path


def check_package(project: Path, output: Path) -> None:
    config = tomllib.loads((project / "pyproject.toml").read_text())
    packages = config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    expected = {}
    source_paths = []
    for package in packages:
        root = project / package
        for source in root.rglob("*.py"):
            expected[source.relative_to(root.parent).as_posix()] = source.read_bytes()
            source_paths.append(source.relative_to(project).as_posix())
    if not expected:
        raise RuntimeError(f"No Python source files found for {project.name}")

    # uv's default build creates an sdist, then builds the wheel from that sdist.
    subprocess.run(["uv", "build", "--out-dir", str(output), str(project)], check=True)
    sdists = list(output.glob("*.tar.gz"))
    wheels = list(output.glob("*.whl"))
    if len(sdists) != 1 or len(wheels) != 1:
        raise RuntimeError("Expected exactly one source distribution and one wheel")
    with tarfile.open(sdists[0]) as archive:
        names = {name.partition("/")[2] for name in archive.getnames()}
        missing = set(source_paths) - names
        if missing:
            raise RuntimeError(f"Source distribution changed or omitted source paths: {sorted(missing)}")
    with zipfile.ZipFile(wheels[0]) as archive:
        code = {name for name in archive.namelist() if name.endswith(".py")}
        if code != set(expected):
            raise RuntimeError(f"Wheel code mismatch: missing={sorted(set(expected) - code)}, extra={sorted(code - set(expected))}")
        for name, content in expected.items():
            if archive.read(name) != content:
                raise RuntimeError(f"Wheel source differs from checkout: {name}")
        if not any(name.endswith("/LICENSE") for name in archive.namelist()):
            raise RuntimeError("Wheel is missing its license")
    print(f"{config['project']['name']} {config['project']['version']}: verified sdist paths and {len(expected)} wheel source files", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        check_package(args.project.resolve(), args.output_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="memorylayer-package-check-") as directory:
            check_package(args.project.resolve(), Path(directory))


if __name__ == "__main__":
    main()
