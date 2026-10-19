#!/usr/bin/env python3
import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

OLD = """        // Frame 1 should still be accessible (LRU order: 1, 2)
        assert!(cache.get(1).is_some());

        // Put frame 3 with 3 bytes (should evict frame 1 due to LRU)
        cache.put(CacheEntry::new(3, vec![3; 3], CacheLocation::RAM)).unwrap();

        // Frame 1 should be evicted, frame 2 should still be there
        assert!(cache.get(1).is_none());
        assert!(cache.get(2).is_some());
"""

NEW = """        // Access frame 1: this promotes it to MRU.
        // LRU order becomes: frame 2 (LRU), frame 1 (MRU).
        assert!(cache.get(1).is_some());

        // Put frame 3 with 3 bytes: frame 2 is now the true LRU and must be evicted.
        cache.put(CacheEntry::new(3, vec![3; 3], CacheLocation::RAM)).unwrap();

        // Frame 1 was recently accessed and must remain; frame 2 must be evicted.
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_none());
"""

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
    if NEW in text:
        print("Correctif déjà présent.")
        return
    if OLD not in text:
        raise SystemExit("Forme du test inconnue: aucun changement appliqué.")

    if not args.apply:
        print("Prévisualisation OK. Le test LRU connu sera corrigé.")
        print("Aucun fichier modifié. Relancer avec --apply --confirm-stopped.")
        return

    if not args.confirm_stopped:
        raise SystemExit("Refus: ajouter --confirm-stopped après avoir arrêté REA.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = workspace / ".rea-cache-fixes" / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup = backup_dir / "lib.rs.before"
    shutil.copy2(target, backup)

    target.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"Correctif LRU appliqué: {target}")
    print(f"Sauvegarde: {backup}")
    print("Vérifier ensuite avec:")
    print("  cd", workspace / "new-app")
    print("  cargo test --offline --locked -p vfx-cache --lib")
    print("  cargo test --offline --locked --workspace --all-targets")

if __name__ == "__main__":
    main()
