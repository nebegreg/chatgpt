#!/usr/bin/env python3
import argparse
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

FUNC_NAME = "cache_budget_replacement_and_lru"

def extract_function_block(text: str, name: str):
    m = re.search(rf"(?m)^\s*fn\s+{re.escape(name)}\s*\([^)]*\)\s*\{{", text)
    if not m:
        return None
    start = m.start()
    brace = text.find("{", m.start(), m.end())
    if brace < 0:
        return None
    depth = 0
    i = brace
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1, text[start:i+1]
        i += 1
    return None

def patch_block(block: str):
    # The current failing test accesses frame 1 before inserting frame 3.
    # In an LRU cache that access promotes frame 1 to MRU, so frame 2 must be evicted.
    p1 = re.compile(r"assert!\s*\(\s*cache\.get\(1\)\.is_none\(\)\s*\)\s*;")
    p2 = re.compile(r"assert!\s*\(\s*cache\.get\(2\)\.is_some\(\)\s*\)\s*;")

    if p1.search(block) is None or p2.search(block) is None:
        return None

    new = p1.sub("assert!(cache.get(1).is_some());", block, count=1)
    new = p2.sub("assert!(cache.get(2).is_none());", new, count=1)

    if new == block:
        return None
    return new

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--confirm-stopped", action="store_true")
    args = ap.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    target = workspace / "new-app" / "crates" / "vfx-cache" / "src" / "lib.rs"
    if not target.is_file():
        raise SystemExit(f"Fichier absent: {target}")

    text = target.read_text(encoding="utf-8")
    found = extract_function_block(text, FUNC_NAME)
    if not found:
        raise SystemExit(f"Test {FUNC_NAME} introuvable: aucun changement appliqué.")

    start, end, block = found

    already_correct = (
        re.search(r"assert!\s*\(\s*cache\.get\(1\)\.is_some\(\)\s*\)\s*;", block)
        and re.search(r"assert!\s*\(\s*cache\.get\(2\)\.is_none\(\)\s*\)\s*;", block)
        and not re.search(r"assert!\s*\(\s*cache\.get\(1\)\.is_none\(\)\s*\)\s*;", block)
    )
    if already_correct:
        print("Correctif LRU déjà présent.")
        return

    patched = patch_block(block)
    if patched is None:
        print("Le test a été trouvé, mais sa forme ne correspond pas aux deux assertions attendues.")
        print("Aucun changement appliqué.")
        print("\nBloc actuel détecté :\n")
        print(block)
        raise SystemExit(2)

    if not args.apply:
        print("Prévisualisation OK.")
        print("Le test cache_budget_replacement_and_lru sera corrigé sans dépendre des commentaires.")
        print("Aucun fichier modifié. Relancer avec --apply --confirm-stopped.")
        return

    if not args.confirm_stopped:
        raise SystemExit("Refus: ajouter --confirm-stopped après avoir arrêté REA.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = workspace / ".rea-cache-fixes" / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup = backup_dir / "lib.rs.before"
    shutil.copy2(target, backup)

    new_text = text[:start] + patched + text[end:]
    target.write_text(new_text, encoding="utf-8")

    print(f"Correctif LRU appliqué: {target}")
    print(f"Sauvegarde: {backup}")
    print("Vérifier ensuite avec:")
    print(f"  cd {workspace / 'new-app'}")
    print("  cargo test --offline --locked -p vfx-cache --lib")
    print("  cargo test --offline --locked --workspace --all-targets")

if __name__ == "__main__":
    main()
