"""Build the final submission zip in the challenge root, laid out as the organisers ask:

    <team>_submission.zip
    ├── output/matching_results.tsv, output/candidate_pairs.tsv
    ├── code/business_entity_resolution/  (src/, README.md, requirements.txt, run_pipeline*.bat)
    └── Documentation_template.md

    python -m src.package --team <team_name>

Caches, trained models, virtual environments and __pycache__ are left out: the code regenerates them.
"""
import argparse
import os
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/business_entity_resolution
ROOT = os.path.dirname(os.path.dirname(HERE))                         # challenge root
CODE_FILES = ("README.md", "requirements.txt", "run_pipeline.bat", "run_pipeline_gpu.bat")


def main():
    ap = argparse.ArgumentParser(description="Zip output/, the code and the documentation for the final submission")
    ap.add_argument("--team", required=True)
    ap.add_argument("--root", default=ROOT, help="challenge root (holds output/ and Documentation_template.md)")
    args = ap.parse_args()

    problems = []
    outputs = [os.path.join(args.root, "output", f) for f in ("matching_results.tsv", "candidate_pairs.tsv")]
    problems += [f"missing {p}" for p in outputs if not os.path.exists(p)]
    doc = os.path.join(args.root, "Documentation_template.md")
    if not os.path.exists(doc):
        problems.append(f"missing {doc}")
    elif "[Your Team Name]" in open(doc, encoding="utf-8").read():
        problems.append(f"{doc} is still the blank template")
    if problems:
        raise SystemExit("[package] not packaging:\n  - " + "\n  - ".join(problems))

    zpath = os.path.join(args.root, f"{args.team}_submission.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in outputs:
            z.write(p, f"output/{os.path.basename(p)}")
        z.write(doc, "Documentation_template.md")
        for f in CODE_FILES:
            z.write(os.path.join(HERE, f), f"code/business_entity_resolution/{f}")
        for dirpath, dirnames, files in os.walk(os.path.join(HERE, "src")):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for f in sorted(files):
                if f.endswith(".py"):
                    full = os.path.join(dirpath, f)
                    z.write(full, "code/business_entity_resolution/" + os.path.relpath(full, HERE).replace(os.sep, "/"))
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
    print(f"[package] {zpath}: {len(names)} files, {os.path.getsize(zpath) / 1e6:.1f} MB")
    for n in names:
        print(f"  {n}")


if __name__ == "__main__":
    main()
