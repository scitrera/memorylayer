#!/usr/bin/env python3
"""Extract API documentation from Python source code using AST parsing.

Reads Python source files and generates Starlight-compatible markdown
with function signatures, docstrings, and type annotations.

Usage:
    python extract_python_api.py <source_dir> <output_file>
"""

import ast
import sys
import textwrap
from pathlib import Path


def extract_class_info(node: ast.ClassDef) -> dict:
    """Extract class name, docstring, and methods."""
    info = {
        "name": node.name,
        "docstring": ast.get_docstring(node) or "",
        "methods": [],
        "bases": [ast.unparse(b) for b in node.bases],
    }

    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name.startswith("_") and item.name != "__init__":
                continue
            method_info = extract_function_info(item)
            info["methods"].append(method_info)

    return info


def extract_function_info(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict:
    """Extract function name, signature, and docstring."""
    is_async = isinstance(node, ast.AsyncFunctionDef)

    args = []
    for arg in node.args.args:
        if arg.arg == "self":
            continue
        annotation = ast.unparse(arg.annotation) if arg.annotation else "Any"
        args.append({"name": arg.arg, "type": annotation})

    # Match defaults to args (defaults are right-aligned)
    defaults = node.args.defaults
    if defaults:
        offset = len(args) - len(defaults)
        for i, default in enumerate(defaults):
            args[offset + i]["default"] = ast.unparse(default)

    return_type = ast.unparse(node.returns) if node.returns else "None"

    return {
        "name": node.name,
        "is_async": is_async,
        "args": args,
        "return_type": return_type,
        "docstring": ast.get_docstring(node) or "",
    }


def format_method_md(method: dict) -> str:
    """Format a method as markdown."""
    prefix = "async " if method["is_async"] else ""
    args_str = ", ".join(
        f"{a['name']}: {a['type']}" + (f" = {a['default']}" if "default" in a else "")
        for a in method["args"]
    )
    sig = f"{prefix}def {method['name']}({args_str}) -> {method['return_type']}"

    lines = [f"#### {method['name']}()", "", f"```python", sig, "```", ""]

    if method["docstring"]:
        lines.append(textwrap.dedent(method["docstring"]).strip())
        lines.append("")

    return "\n".join(lines)


def format_class_md(cls: dict) -> str:
    """Format a class as markdown."""
    lines = [f"## {cls['name']}", ""]

    if cls["docstring"]:
        lines.append(cls["docstring"].strip())
        lines.append("")

    for method in cls["methods"]:
        lines.append(format_method_md(method))

    return "\n".join(lines)


def extract_from_file(filepath: Path) -> list[dict]:
    """Extract classes and functions from a Python file."""
    source = filepath.read_text()
    tree = ast.parse(source)
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            classes.append(extract_class_info(node))

    return classes


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <source_dir> <output_file>")
        sys.exit(1)

    source_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        sys.exit(1)

    all_classes = []
    for py_file in sorted(source_dir.glob("*.py")):
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue
        classes = extract_from_file(py_file)
        all_classes.extend(classes)

    # Generate markdown
    lines = [
        "---",
        "title: Python API Reference",
        "description: Auto-generated API reference for the MemoryLayer Python SDK",
        "---",
        "",
        ":::note",
        "This reference is auto-generated from source code. See the [Quick Start](/sdk-python/quickstart/) for usage examples.",
        ":::",
        "",
    ]

    for cls in all_classes:
        lines.append(format_class_md(cls))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines))
    print(f"Generated: {output_file}")


if __name__ == "__main__":
    main()
