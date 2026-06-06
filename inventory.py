#!/usr/bin/env python3
"""Inventory: per photo, check local availability of original + preview derivatives."""
import csv
import os
import sys
from pathlib import Path

import osxphotos

OUT_CSV = Path.home() / "photo-sort" / "metadata" / "inventory.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

print("Loading PhotosDB...", flush=True)
db = osxphotos.PhotosDB()
photos = db.photos(intrash=False)
print(f"Total photos (not in trash): {len(photos)}", flush=True)

stats = {
    "total": 0,
    "has_local_original": 0,
    "has_local_derivative": 0,
    "derivative_total_bytes": 0,
    "no_local_anything": 0,
}

with OUT_CSV.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "uuid", "original_filename", "date", "ismissing",
        "has_local_original", "has_local_derivative",
        "derivative_path", "derivative_size_bytes",
        "uti_original", "isphoto", "ismovie",
    ])
    for i, p in enumerate(photos):
        stats["total"] += 1
        has_orig = (p.path is not None) and os.path.exists(p.path)
        derivs = p.path_derivatives or []
        deriv_path = ""
        deriv_size = 0
        if derivs:
            # path_derivatives is sorted largest-first; pick largest
            for d in derivs:
                if d and os.path.exists(d):
                    try:
                        sz = os.path.getsize(d)
                        if sz > deriv_size:
                            deriv_size = sz
                            deriv_path = d
                    except OSError:
                        pass
        has_deriv = bool(deriv_path)

        if has_orig:
            stats["has_local_original"] += 1
        if has_deriv:
            stats["has_local_derivative"] += 1
            stats["derivative_total_bytes"] += deriv_size
        if not has_orig and not has_deriv:
            stats["no_local_anything"] += 1

        w.writerow([
            p.uuid, p.original_filename or "",
            p.date.isoformat() if p.date else "",
            p.ismissing, has_orig, has_deriv,
            deriv_path, deriv_size,
            p.uti_original or "", p.isphoto, p.ismovie,
        ])
        if (i + 1) % 5000 == 0:
            print(f"  ...processed {i+1}", flush=True)

print("\n=== INVENTORY ===")
print(f"Total:                 {stats['total']}")
print(f"Local original:        {stats['has_local_original']} ({stats['has_local_original']*100/stats['total']:.1f}%)")
print(f"Local derivative:      {stats['has_local_derivative']} ({stats['has_local_derivative']*100/stats['total']:.1f}%)")
print(f"No local anything:     {stats['no_local_anything']} ({stats['no_local_anything']*100/stats['total']:.1f}%)")
print(f"Derivative size total: {stats['derivative_total_bytes']/1e9:.2f} GB")
print(f"CSV written:           {OUT_CSV}")
