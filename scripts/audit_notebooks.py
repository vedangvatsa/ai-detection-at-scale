#!/usr/bin/env python3
"""
Audit Kaggle notebooks (.ipynb) for stale imports, hardcoded paths, and other slop.

Usage:
    python scripts/audit_notebooks.py
"""
import os
import sys
import json
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
NOTEBOOKS_DIR = os.path.join(PROJECT_DIR, 'notebooks')

HARDcoded_PATH_RE = re.compile(r'/Users/\w+|/home/\w+|C:\\\\Users\\\\\w+|/kaggle/working')
TOOL_IMPORT_RE = re.compile(r'(?:from|import)\s+tool\b')
SCRIPT_IMPORT_RE = re.compile(r'(?:from|install)\s+scripts\b')
BANG_SCRIPT_RE = re.compile(r'!python\s+scripts/(\S+)')


def collect_notebooks(root):
    notebooks = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.ipynb'):
                notebooks.append(os.path.join(dirpath, fn))
    return notebooks


def audit_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    issues = []
    cells = nb.get('cells', [])
    for idx, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue
        source = ''.join(cell.get('source', []))
        if HARDcoded_PATH_RE.search(source):
            issues.append((idx, 'hardcoded absolute path', HARDcoded_PATH_RE.findall(source)))
        if TOOL_IMPORT_RE.search(source):
            # Verify the referenced module exists.
            for mod in re.findall(r'(?:from|import)\s+tool\.(\w+)', source):
                mod_path = os.path.join(PROJECT_DIR, 'tool', f'{mod}.py')
                if not os.path.exists(mod_path):
                    issues.append((idx, f'missing tool module: tool.{mod}', None))
        if SCRIPT_IMPORT_RE.search(source):
            for mod in re.findall(r'(?:from|import)\s+scripts\.(\w+)', source):
                script_path = os.path.join(PROJECT_DIR, 'scripts', f'{mod}.py')
                if not os.path.exists(script_path):
                    issues.append((idx, f'missing script module: scripts.{mod}', None))
        for script in BANG_SCRIPT_RE.findall(source):
            script_path = os.path.join(PROJECT_DIR, script)
            if not os.path.exists(script_path):
                issues.append((idx, f'missing bang script: {script}', None))
    return issues


def main():
    notebooks = collect_notebooks(NOTEBOOKS_DIR)
    print(f'Auditing {len(notebooks)} notebooks...\n')
    any_issue = False
    for nb_path in sorted(notebooks):
        rel = os.path.relpath(nb_path, PROJECT_DIR)
        issues = audit_notebook(nb_path)
        if issues:
            any_issue = True
            print(f'[ISSUES] {rel}')
            for idx, reason, detail in issues:
                print(f'  cell {idx}: {reason}', end='')
                if detail:
                    print(f' -> {detail}')
                else:
                    print()
        else:
            print(f'[OK]     {rel}')
    print()
    sys.exit(1 if any_issue else 0)


if __name__ == '__main__':
    main()
