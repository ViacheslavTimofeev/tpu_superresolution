# split_train_test_pairs.py
from __future__ import annotations
import argparse, random, re, shutil
from pathlib import Path
from typing import Dict, List, Tuple

RX_Z   = re.compile(r"_z(\d{4})\.(?:tif|tiff|png)$", re.IGNORECASE)
RX_G   = re.compile(r"_g(\d{4})\.(?:tif|tiff|png)$", re.IGNORECASE)
EXTS   = {".tif", ".tiff", ".png"}

def index_lr(dir_lr: Path) -> Dict[int, Path]:
    idx = {}
    for p in dir_lr.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS:
            m = RX_Z.search(p.name)
            if m: idx[int(m.group(1))] = p
    return idx

def index_hr(dir_hr: Path) -> Dict[int, Path]:
    """
    Предпочитаем глобальный индекс из _gNNNN.
    Если его нет, падёмся на _zNNNN (как «глобальный» ключ).
    """
    idx_g, idx_z = {}, {}
    for p in dir_hr.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS:
            mg = RX_G.search(p.name)
            if mg: idx_g[int(mg.group(1))] = p
            else:
                mz = RX_Z.search(p.name)
                if mz: idx_z[int(mz.group(1))] = p
    return idx_g if idx_g else idx_z

def parse_test_size(s: str, total: int) -> int:
    try:
        v = float(s)
        if 0 < v <= 1:  # доля
            return max(1, int(round(total * v)))
        v_int = int(v)
        if 1 <= v_int <= total:
            return v_int
    except Exception:
        pass
    raise ValueError(f"--test-size '{s}' некорректен для total={total}. Примеры: 0.2 или 200")

def main():
    ap = argparse.ArgumentParser("Random split согласованных пар LR/HR в train/test")
    ap.add_argument("--root", type=Path, required=True, help="Корень с папками LR_train, HR_train")
    ap.add_argument("--test-size", type=str, required=True, help="Доля (0..1] или число пар (int)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="Перемещать (по умолчанию копировать)")
    ap.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано")
    args = ap.parse_args()

    lr_train = args.root / "LR_train"
    hr_train = args.root / "HR_train"
    lr_test  = args.root / "LR_test"
    hr_test  = args.root / "HR_test"
    for d in (lr_train, hr_train):
        if not d.is_dir():
            raise FileNotFoundError(f"Не найдена папка: {d}")

    # Индексация
    lr_idx = index_lr(lr_train)
    hr_idx = index_hr(hr_train)
    keys = sorted(set(lr_idx.keys()) & set(hr_idx.keys()))
    if not keys:
        raise RuntimeError("Не найдено пересечения ключей LR и HR. Проверь имена файлов (_zNNNN/_gNNNN).")

    n_total = len(keys)
    n_test = parse_test_size(args.test_size, n_total)

    random.seed(args.seed)
    test_keys = set(random.sample(keys, n_test))

    # Создать выходные папки
    lr_test.mkdir(parents=True, exist_ok=True)
    hr_test.mkdir(parents=True, exist_ok=True)

    op = shutil.move if args.move else shutil.copy2

    # Выполнить перенос/копирование
    plan: List[Tuple[Path, Path]] = []
    for k in test_keys:
        plan.append((lr_idx[k], lr_test / lr_idx[k].name))
        plan.append((hr_idx[k], hr_test / hr_idx[k].name))

    print(f"Всего пар в train: {n_total}")
    print(f"В test уйдёт пар:   {n_test} ({'move' if args.move else 'copy'})")
    if args.dry_run:
        for src, dst in plan[:10]:
            print("[dry-run]", src, "->", dst)
        print("... (показаны первые 10 действий)")
        return

    for src, dst in plan:
        op(str(src), str(dst))

    print("Готово.")
    if args.move:
        print("Файлы перемещены. Остаток в train уменьшился.")
    else:
        print("Файлы скопированы. Train остался нетронутым.")

if __name__ == "__main__":
    main()
