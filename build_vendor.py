"""
Build vendor/ for the text-only TRELLIS Modly extension.

This script vendors only pure-Python runtime sources:
  - official microsoft/TRELLIS trellis/ package slices needed by text-to-3D
  - official utils3d fork used by TRELLIS postprocessing
  - small pure-Python helper packages when convenient

It does not vendor native CUDA packages. setup.py owns native installation.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


VENDOR = Path(__file__).parent / "vendor"
TRELLIS_REF = "442aa1e1afb9014e80681d3bf604e8d728a86ee7"
TRELLIS_ZIP = f"https://github.com/microsoft/TRELLIS/archive/{TRELLIS_REF}.zip"
UTILS3D_REF = "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
FLEXICUBES_SUBMODULE_PATH = "trellis/representations/mesh/flexicubes"
TEXT_ONLY_VENDOR_MARKER = ".trellis-text-only-v3"

PURE_PACKAGES = [
    "easydict",
    "einops",
    "tqdm",
]

FORBIDDEN_VENDOR_DIRS = {
    "nvdiffrast",
    "diff_gaussian_rasterization",
    "spconv",
    "cumm",
    "cumesh",
    "o_voxel",
    "trellis2",
}


def run(cmd: list[str], **kwargs) -> None:
    print(f"  $ {' '.join(str(part) for part in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def vendor_pure_package(package: str, dest: Path) -> None:
    run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(dest), "--upgrade", package])


def vendor_utils3d(dest: Path) -> None:
    run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(dest), "--upgrade", UTILS3D_REF])


def clean_forbidden_vendor_dirs(dest: Path) -> None:
    for name in sorted(FORBIDDEN_VENDOR_DIRS):
        path = dest / name
        if path.exists():
            print(f"  Removing forbidden/native vendor directory: {path}")
            shutil.rmtree(path)


def vendor_trellis(dest: Path) -> None:
    trellis_dest = dest / "trellis"
    if trellis_dest.exists():
        print("  Removing existing trellis/ before refreshing official source slice.")
        shutil.rmtree(trellis_dest)

    print("  Downloading official TRELLIS source archive...")
    with urllib.request.urlopen(TRELLIS_ZIP, timeout=180) as resp:
        data = resp.read()

    archive_root = f"TRELLIS-{TRELLIS_REF}/"
    allowed_prefixes = [
        f"{archive_root}trellis/__init__.py",
        f"{archive_root}trellis/models/",
        f"{archive_root}trellis/modules/",
        f"{archive_root}trellis/pipelines/",
        f"{archive_root}trellis/renderers/",
        f"{archive_root}trellis/representations/",
        f"{archive_root}trellis/utils/",
    ]

    extracted = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if not any(member.startswith(prefix) for prefix in allowed_prefixes):
                continue
            rel = member[len(archive_root):]
            target = dest / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
                extracted += 1

    if extracted == 0:
        raise RuntimeError("No official trellis/ files were extracted. Check TRELLIS_REF/archive layout.")
    print(f"  trellis/ extracted ({extracted} files).")
    sync_trellis_runtime_submodules(dest)
    patch_trellis_text_only_exports(dest)
    patch_trellis_text_pipeline_optional_open3d(dest)


def patch_trellis_text_only_exports(dest: Path) -> None:
    """Prevent image-only TRELLIS pipelines from being imported in text-only runtime.

    Official TRELLIS exposes every pipeline from `trellis/pipelines/__init__.py`.
    Importing that module pulls the image-to-3D pipeline, which imports `rembg`.
    This extension intentionally does not ship image preprocessing dependencies,
    so the vendored pipeline package must export only the native text pipeline.
    """

    pipelines_dir = dest / "trellis" / "pipelines"
    pipelines_init = pipelines_dir / "__init__.py"
    text_pipeline = pipelines_dir / "trellis_text_to_3d.py"
    if not text_pipeline.exists():
        raise RuntimeError(f"Missing TRELLIS text pipeline file: {text_pipeline}")
    pipelines_init.write_text(
        "\"\"\"Text-only TRELLIS pipeline exports for Modly.\"\"\"\n"
        "\n"
        "from .trellis_text_to_3d import TrellisTextTo3DPipeline\n"
        "\n"
        "__all__ = [\"TrellisTextTo3DPipeline\"]\n",
        encoding="utf-8",
    )
    print("  Patched trellis/pipelines/__init__.py for text-only exports.")


def patch_trellis_text_pipeline_optional_open3d(dest: Path) -> None:
    """Make open3d optional for the prompt-to-mesh text pipeline.

    Official `trellis_text_to_3d.py` imports open3d at module import time, but
    open3d is only used by `voxelize()`/`run_variant()` for mesh-conditioned
    variants. The Modly text-only node uses `run(prompt=...)`, so requiring
    open3d during module import is unnecessary and breaks platforms where
    open3d wheels are unavailable.
    """

    pipeline_path = dest / "trellis" / "pipelines" / "trellis_text_to_3d.py"
    source = pipeline_path.read_text(encoding="utf-8")
    source = source.replace("import open3d as o3d\n", "")
    source = source.replace("mesh: o3d.geometry.TriangleMesh", "mesh: Any")
    source = source.replace(
        "    def voxelize(self, mesh: Any) -> torch.Tensor:\n",
        "    def voxelize(self, mesh: Any) -> torch.Tensor:\n"
        "        try:\n"
        "            import open3d as o3d\n"
        "        except ImportError as exc:\n"
        "            raise RuntimeError(\n"
        "                \"open3d is required only for TRELLIS text run_variant()/mesh-conditioned variants. \"\n"
        "                \"The Modly text-to-mesh node does not use this path.\"\n"
        "            ) from exc\n",
        1,
    )
    pipeline_path.write_text(source, encoding="utf-8")
    print("  Patched trellis_text_to_3d.py so open3d is optional for prompt-to-mesh runtime.")


def write_text_only_vendor_marker(dest: Path) -> None:
    (dest / TEXT_ONLY_VENDOR_MARKER).write_text(
        "text-only TRELLIS vendor prepared by build_vendor.py; image exports disabled; open3d lazy-loaded; kaolin removed\n",
        encoding="utf-8",
    )


def trellis_submodule_ref(path: str) -> tuple[str, str]:
    import json

    url = f"https://api.github.com/repos/microsoft/TRELLIS/contents/{path}?ref={TRELLIS_REF}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        metadata = json.load(resp)
    if metadata.get("type") != "submodule":
        raise RuntimeError(f"Expected '{path}' to be a TRELLIS submodule at ref {TRELLIS_REF}.")
    repo_url = metadata.get("submodule_git_url")
    commit = metadata.get("sha")
    if not repo_url or not commit:
        raise RuntimeError(f"Missing submodule metadata for '{path}'.")
    return repo_url, commit


def vendor_flexicubes_submodule(dest: Path) -> None:
    repo_url, commit = trellis_submodule_ref(FLEXICUBES_SUBMODULE_PATH)
    archive_url = f"{repo_url[:-4]}/archive/{commit}.zip" if repo_url.endswith(".git") else f"{repo_url}/archive/{commit}.zip"
    package_dest = dest / FLEXICUBES_SUBMODULE_PATH
    package_dest.mkdir(parents=True, exist_ok=True)

    print(f"  Syncing FlexiCubes submodule from {repo_url} @ {commit}...")
    with urllib.request.urlopen(archive_url, timeout=180) as resp:
        data = resp.read()

    archive_root = None
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if archive_root is None:
                archive_root = member.split("/", 1)[0] + "/"
            if member in {f"{archive_root}flexicubes.py", f"{archive_root}tables.py"}:
                target = package_dest / member[len(archive_root):]
                target.write_bytes(zf.read(member))

    expected = [package_dest / "flexicubes.py", package_dest / "tables.py"]
    missing = [path.name for path in expected if not path.exists()]
    if missing:
        raise RuntimeError("Failed to vendor FlexiCubes runtime files: " + ", ".join(missing))
    (package_dest / "__init__.py").write_text("from .flexicubes import FlexiCubes\n", encoding="utf-8")
    patch_flexicubes_local_check_tensor(package_dest)


def patch_flexicubes_local_check_tensor(package_dest: Path) -> None:
    """Remove kaolin as a hard dependency from vendored FlexiCubes.

    Upstream FlexiCubes imports `kaolin.utils.testing.check_tensor` only for
    shape assertions. Pulling full kaolin into this extension would add a heavy
    native dependency for a tiny helper, so we provide an equivalent local
    checker for the FlexiCubes usage pattern.
    """

    flexicubes = package_dest / "flexicubes.py"
    source = flexicubes.read_text(encoding="utf-8")
    kaolin_import = "from kaolin.utils.testing import check_tensor\n"
    replacement = '''\n\ndef check_tensor(tensor, shape=None, dtype=None, device=None, throw=True):\n    ok = torch.is_tensor(tensor)\n    if ok and shape is not None:\n        ok = len(tensor.shape) == len(shape) and all(expected is None or actual == expected for actual, expected in zip(tensor.shape, shape))\n    if ok and dtype is not None:\n        ok = tensor.dtype == dtype\n    if ok and device is not None:\n        ok = tensor.device == device\n    if throw and not ok:\n        raise AssertionError(f\"Invalid tensor: shape={getattr(tensor, 'shape', None)}, dtype={getattr(tensor, 'dtype', None)}, device={getattr(tensor, 'device', None)}\")\n    return ok\n'''
    if kaolin_import not in source:
        raise RuntimeError("Expected kaolin check_tensor import was not found in FlexiCubes source.")
    flexicubes.write_text(source.replace(kaolin_import, replacement, 1), encoding="utf-8")
    print("  Patched FlexiCubes to use a local check_tensor helper instead of kaolin.")


def sync_trellis_runtime_submodules(dest: Path) -> None:
    vendor_flexicubes_submodule(dest)


def main() -> None:
    print(f"Building text-only vendor/ in {VENDOR}")
    VENDOR.mkdir(parents=True, exist_ok=True)
    clean_forbidden_vendor_dirs(VENDOR)

    print("\n[1] Vendoring small pure-Python helper packages...")
    for package in PURE_PACKAGES:
        try:
            vendor_pure_package(package, VENDOR)
        except Exception as exc:
            print(f"  WARNING: failed to vendor {package}: {exc}")

    print("\n[2] Vendoring official utils3d fork...")
    vendor_utils3d(VENDOR)

    print("\n[3] Vendoring official TRELLIS text runtime source...")
    vendor_trellis(VENDOR)

    clean_forbidden_vendor_dirs(VENDOR)
    write_text_only_vendor_marker(VENDOR)
    print("\nDone. vendor/ contains pure-Python TRELLIS text runtime assets only.")


if __name__ == "__main__":
    main()
