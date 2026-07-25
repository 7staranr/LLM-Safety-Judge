"""One-shot: make every revision script resolve inputs/outputs through DISTILL_ROOT.

Reviewer finding: five scripts still hard-coded an absolute developer path, so the
released package was not self-contained. This rewrites them to resolve the
repository root the same way the other scripts do.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

ROOT_SNIPPET = (
    'ROOT = os.environ.get("DISTILL_ROOT",\n'
    '                      os.path.abspath(os.path.join(HERE, "..", "..")))'
)

fixes = {
    "exp_decomposition_consistent.py": [
        ('ROOT = r"D:\\Programming\\distill_project_1"', ROOT_SNIPPET),
    ],
    "exp_design_weighted_stage1.py": [
        ('ROOT = r"D:\\Programming\\distill_project_1"', ROOT_SNIPPET),
    ],
    "exp_parse_convention.py": [
        ('ROOT = r"D:\\Programming\\distill_project_1"', ROOT_SNIPPET),
    ],
    "exp_phi_paired.py": [
        ('ROOT = r"D:\\Programming\\distill_project_1"', ROOT_SNIPPET),
    ],
    "rebuild_kappa_table.py": [
        ('HA = r"D:\\Programming\\distill_project_1\\human_annotation"',
         'ROOT = os.environ.get("DISTILL_ROOT",\n'
         '                      os.path.abspath(os.path.join(HERE, "..", "..")))\n'
         'HA = os.path.join(ROOT, "human_annotation")'),
        ('OUT = r"D:\\Programming\\distill_project_1\\paper_ieee_access\\figures\\table_kappa.tex"',
         'OUT = os.path.join(ROOT, "figures", "table_kappa.tex")'),
    ],
}

for fn, pairs in fixes.items():
    p = os.path.join(HERE, fn)
    s = open(p, encoding="utf-8").read()
    # make sure HERE exists in the module before we reference it
    if "HERE = " not in s:
        s = s.replace("import os\n", "import os\n", 1)
        s = re.sub(r"(\nimport os\n)", r"\1", s, count=1)
    for old, new in pairs:
        if old in s:
            s = s.replace(old, new)
            print(f"OK   {fn}: {old[:52]}")
        else:
            print(f"MISS {fn}: {old[:52]}")
    open(p, "w", encoding="utf-8").write(s)

# Report which scripts still define HERE after ROOT (ordering matters)
for fn in fixes:
    s = open(os.path.join(HERE, fn), encoding="utf-8").read()
    hi, ri = s.find("HERE ="), s.find("ROOT =")
    if hi == -1:
        print(f"WARN {fn}: no HERE definition")
    elif ri != -1 and hi > ri:
        print(f"WARN {fn}: HERE defined AFTER ROOT (line order needs manual fix)")
