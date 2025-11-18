#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Подготовка общего LR и последовательных HR-блоков (ILS1/ILS2/ILS3) из .mat в float32 TIFF,
с учётом глобального Z и смещений блоков.

Структура вывода:
<out_root>/
  LR/                       block_LR_z0000.tif, block_LR_z0001.tif, ...
  ILS1/HR/                  block_ILS1_z0000_g0000.tif, ...
  ILS2/HR/                  block_ILS2_z0000_gNNNN.tif, ...
  ILS3/HR/                  ...
  manifest.json

Запуск:
  python prepare_mrccm.py \
    --lr_mat path/to/ILS_LR.mat \
    --hr ILS1=path/to/ILS1_HR.mat \
    --hr ILS2=path/to/ILS2_HR.mat \
    --hr ILS3=path/to/ILS3_HR.mat \
    --out_root MRCCM2D \
    --force_4x \
    --workers 8 --chunk 16
"""

import argparse, os, json, time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm

# ---------- utils ----------

def find_first_dataset_key(f: h5py.File) -> str:
    for k, v in f.items():
        if isinstance(v, h5py.Dataset):
            return k
    def walk(g):
        for kk, vv in g.items():
            if isinstance(vv, h5py.Dataset):
                return (g.name.strip("/") + "/" + kk) if g.name != "/" else kk
            if isinstance(vv, h5py.Group):
                r = walk(vv)
                if r: return r
        return None
    k = walk(f)
    if not k:
        raise KeyError("В .mat не найден ни один h5py.Dataset")
    return k

def save_float32_tiff(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import tifffile as tiff
        tiff.imwrite(str(path), arr.astype(np.float32, copy=False),
                     dtype=np.float32, bigtiff=True, compression=None)
    except Exception:
        Image.fromarray(arr.astype(np.float32), mode="F").save(str(path))

def write_manifest(manifest_path: Path, payload: dict, overwrite: bool):
    if manifest_path.exists() and not overwrite:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (old.get("lr_mat") != payload.get("lr_mat")) or (old.get("hr_mats") != payload.get("hr_mats")):
            raise RuntimeError(
                "manifest.json уже существует и указывает другие источники.\n"
                "Добавь --overwrite, если намеренно перезаписываешь."
            )
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

# ---------- mapping offsets ----------

def parse_hr_args(hr_list: List[str]) -> Dict[str, Path]:
    """--hr NAME=path → dict(Name -> Path), порядок важен (последовательность вдоль керна)."""
    result = {}
    for item in hr_list:
        if "=" not in item:
            raise argparse.ArgumentTypeError("Каждый --hr должен быть в виде NAME=PATH")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise argparse.ArgumentTypeError("Имя блока в --hr пустое")
        result[name] = Path(path.strip())
    return result

def parse_hr_keys(hr_key_list: List[str]) -> Dict[str, Optional[str]]:
    """--hr_key NAME=KEY (необязательно для каждого блока)."""
    out: Dict[str, Optional[str]] = {}
    for item in hr_key_list or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError("Каждый --hr_key должен быть в виде NAME=KEY")
        name, key = item.split("=", 1)
        out[name.strip()] = key.strip()
    return out

def compute_layout(
    lr_mat: Path, lr_key: Optional[str],
    hr_mats: Dict[str, Path], hr_keys: Dict[str, Optional[str]],
    force_4x: bool, start_z: Optional[int], end_z: Optional[int]
) -> Tuple[int, int, List[str], Dict[str, int], Dict[str, int],
           Dict[str, Tuple[int,int,int]], Tuple[int,int,int], int]:
    """
    Возвращает:
      z0, z1_global, order, capacities_lr, offsets_lr, hr_shapes, lr_shape, nz_lr
    где:
      capacities_lr[B] — ёмкость блока B в «LR-срезах» (floor(nz_hr/4) при force_4x, иначе nz_hr)
      offsets_lr[B]    — глобальное смещение LR для блока B (в LR-срезах), кумулятивно по order
      z1_global        — min(nz_lr, sum(capacities_lr.values())) и обрезан по end_z, если задан
    """
    # LR shape
    with h5py.File(lr_mat, "r") as flr:
        lrk = lr_key or find_first_dataset_key(flr)
        LR = flr[lrk]
        lr_shape = (int(LR.shape[0]), int(LR.shape[1]), int(LR.shape[2]))
        nz_lr = lr_shape[0]

    # HR shapes + capacities
    order = list(hr_mats.keys())  # порядок вдоль керна == порядок аргументов
    hr_shapes = {}
    capacities_lr: Dict[str, int] = {}
    for name in order:
        with h5py.File(hr_mats[name], "r") as fhr:
            hrk = hr_keys.get(name) or find_first_dataset_key(fhr)
            HR = fhr[hrk]
            shape = (int(HR.shape[0]), int(HR.shape[1]), int(HR.shape[2]))
            hr_shapes[name] = shape
            cap = shape[0] // 4 if force_4x else shape[0]
            capacities_lr[name] = int(cap)

    # offsets
    offsets_lr: Dict[str, int] = {}
    acc = 0
    for name in order:
        offsets_lr[name] = acc
        acc += capacities_lr[name]

    # global z range
    z0 = 0 if start_z is None else max(0, int(start_z))
    z1_by_hr = acc  # sum of capacities
    z1_req = nz_lr if end_z is None else min(nz_lr, int(end_z))
    z1_global = min(nz_lr, z1_by_hr, z1_req)
    if z1_global <= z0:
        raise ValueError(f"Пустой диапазон z: z0={z0}, z1={z1_global} (учёт LR и всех HR, force_4x={force_4x})")

    return z0, z1_global, order, capacities_lr, offsets_lr, hr_shapes, lr_shape, nz_lr

def block_for_global_z(z_global: int, order: List[str], offsets: Dict[str,int], capacities: Dict[str,int]) -> Tuple[str,int]:
    """
    По глобальному z возвращает (имя блока, локальный z) такой, что offset[b] <= z < offset[b]+capacity[b].
    """
    # Поскольку блоков немного (1–3), линейный поиск дешевле и проще.
    for name in order:
        z0 = offsets[name]; cap = capacities[name]
        if z0 <= z_global < (z0 + cap):
            return name, z_global - z0
    raise IndexError(f"z_global={z_global} не попал ни в один блок (offsets/capacities неконсистентны)")

# ---------- worker ----------

def _worker_chunk(
    lr_mat_path: str, lr_key: Optional[str],
    hr_mats: Dict[str, str], hr_keys: Dict[str, Optional[str]],
    out_root: str, fmt: str, force_4x: bool,
    zlist_global: List[int],
    order: List[str], capacities: Dict[str,int], offsets: Dict[str,int]
):
    import h5py, numpy as np
    from pathlib import Path
    written = 0; processed = 0
    root = Path(out_root); (root / "LR").mkdir(parents=True, exist_ok=True)

    # открываем LR
    with h5py.File(lr_mat_path, "r") as flr:
        lrk = lr_key or _first_key(flr := flr)
        LR = flr[lrk]

        # открываем все HR-блоки (мало файлов — можно держать открытыми)
        fhrs = {}; HRs = {}
        for bname in order:
            fh = h5py.File(hr_mats[bname], "r")
            hrk = hr_keys.get(bname) or _first_key(fh)
            fhrs[bname] = fh; HRs[bname] = fh[hrk]

        try:
            for zg in zlist_global:
                # LR (общий)
                lr2d = np.array(LR[zg, :, :], dtype=np.float32)
                lr_fn = f"block_LR_z{zg:04d}.{fmt}"
                lr_path = root / "LR" / lr_fn
                if not lr_path.exists():
                    save_float32_tiff(lr2d, lr_path)
                    written += lr2d.nbytes

                # Выбираем соответствующий HR-блок
                bname, z_local = block_for_global_z(zg, order, offsets, capacities)
                HR = HRs[bname]

                # HR slice
                if force_4x:
                    base = 4 * z_local
                    if base + 3 >= HR.shape[0]:
                        # теоретически не должно случиться из-за capacities; защищаемся
                        continue
                    sls = [np.array(HR[base + k, :, :], dtype=np.float32) for k in range(4)]
                    hr2d = np.mean(np.stack(sls, 0), 0).astype(np.float32, copy=False)
                else:
                    if z_local >= HR.shape[0]:
                        continue
                    hr2d = np.array(HR[z_local, :, :], dtype=np.float32)

                out_hr_dir = root / bname / "HR"
                out_hr_dir.mkdir(parents=True, exist_ok=True)
                # ВАЖНО: в имени HR кладём и локальный, и глобальный индексы
                hr_fn = f"block_{bname}_z{z_local:04d}_g{zg:04d}.{fmt}"
                save_float32_tiff(hr2d, out_hr_dir / hr_fn)
                written += hr2d.nbytes

                processed += 1
        finally:
            for fh in fhrs.values():
                try: fh.close()
                except: pass

    return written, processed

def _first_key(f):
    for k,v in f.items():
        if isinstance(v, h5py.Dataset):
            return k
    def walk(g):
        for kk, vv in g.items():
            if isinstance(vv, h5py.Dataset):
                return (g.name.strip('/') + '/' + kk) if g.name != '/' else kk
            if isinstance(vv, h5py.Group):
                r = walk(vv)
                if r: return r
        return None
    r = walk(f)
    if not r: raise KeyError("В .mat не найден ни один h5py.Dataset")
    return r

# ---------- main ----------

def convert_all(
    lr_mat: Path, hr_mats: Dict[str, Path], out_root: Path,
    lr_key: Optional[str], hr_keys: Dict[str, Optional[str]],
    force_4x: bool, fmt: str,
    start_z: Optional[int], end_z: Optional[int],
    workers: int, chunk: int,
    overwrite: bool
):
    out_root.mkdir(parents=True, exist_ok=True)

    z0, z1_global, order, capacities, offsets, hr_shapes, lr_shape, nz_lr = compute_layout(
        lr_mat, lr_key, hr_mats, hr_keys, force_4x, start_z, end_z
    )

    manifest = {
        "lr_mat": str(lr_mat.resolve()),
        "hr_mats": {k: str(v.resolve()) for k, v in hr_mats.items()},
        "force_4x": bool(force_4x),
        "format": fmt,
        "order": order,
        "capacities_lr": capacities,
        "offsets_lr": offsets,
        "z_range_effective_global": [int(z0), int(z1_global)],
        "lr_shape": list(lr_shape),
        "hr_shapes": {k: list(v) for k, v in hr_shapes.items()},
        "note": "HR имена содержат локальный и глобальный индексы: ..._zLLLL_gGGGG.tif; LR — по глобальному z.",
    }
    write_manifest(out_root / "manifest.json", manifest, overwrite=overwrite)

    for bname in order:
        (out_root / bname / "HR").mkdir(parents=True, exist_ok=True)
    (out_root / "LR").mkdir(parents=True, exist_ok=True)

    indices_global = list(range(z0, z1_global))
    chunks = [indices_global[i:i+chunk] for i in range(0, len(indices_global), chunk)]
    t0 = time.time(); total = len(indices_global); bytes_written = 0

    with ProcessPoolExecutor(max_workers=workers) as ex, tqdm(total=total, desc="MRCCM", unit="slice") as pbar:
        futures = []
        for zlist in chunks:
            futures.append(
                ex.submit(
                    _worker_chunk,
                    str(lr_mat), lr_key,
                    {k: str(v) for k,v in hr_mats.items()},
                    hr_keys,
                    str(out_root), fmt, force_4x,
                    zlist, order, capacities, offsets
                )
            )
        for fut in as_completed(futures):
            w, n = fut.result()
            bytes_written += w
            pbar.update(n)
            elapsed = time.time() - t0
            if elapsed > 0:
                rate = pbar.n / elapsed
                mbps = (bytes_written / (1024**2)) / elapsed
                pbar.set_postfix_str(f"{rate:.1f} sl/s, {mbps:.1f} MB/s")

    print("[done] Saved under:", out_root.resolve())
    print("  LR:  ", (out_root / "LR").resolve())
    for b in order:
        print(f"  {b}/HR:", (out_root / b / "HR").resolve())
    print("Manifest:", (out_root / 'manifest.json').resolve())

def parse_args():
    ap = argparse.ArgumentParser("Prepare MRCCM (.mat) → shared LR + sequential multi-block HR (float32 TIFF)")
    ap.add_argument("--lr_mat", type=Path, required=True, help="Путь к общему LR .mat")
    ap.add_argument("--lr_key", type=str, default=None, help="Ключ датасета в LR .mat (если не указан — авто)")
    ap.add_argument("--hr", action="append", required=True,
                    help="Указывать несколько раз в ПРАВИЛЬНОМ порядке: --hr ILS1=path --hr ILS2=path ...")
    ap.add_argument("--hr_key", action="append", default=[],
                    help="Необязательно: соответствующие ключи для HR, формат NAME=KEY")
    ap.add_argument("--out_root", type=Path, required=True, help="Корень вывода")
    ap.add_argument("--force_4x", action="store_true", help="HR[z_local] = mean(HR[4*z_local..+3])")
    ap.add_argument("--format", type=str, choices=["tif","tiff"], default="tiff", help="float32 TIFF")
    ap.add_argument("--start_z", type=int, default=None, help="Начальный глобальный индекс Z (включительно)")
    ap.add_argument("--end_z", type=int, default=None, help="Конечный глобальный индекс Z (исключительно)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Процессов чтения/записи")
    ap.add_argument("--chunk", type=int, default=8, help="Число глобальных срезов на один процесс")
    ap.add_argument("--overwrite", action="store_true", help="Переписать manifest.json, если отличается")
    return ap.parse_args()

def main():
    args = parse_args()
    hr_mats = parse_hr_args(args.hr)
    hr_keys = parse_hr_keys(args.hr_key)
    convert_all(
        lr_mat=args.lr_mat, hr_mats=hr_mats, out_root=args.out_root,
        lr_key=args.lr_key, hr_keys=hr_keys,
        force_4x=args.force_4x, fmt=args.format,
        start_z=args.start_z, end_z=args.end_z,
        workers=args.workers, chunk=args.chunk,
        overwrite=args.overwrite
    )

if __name__ == "__main__":
    main()