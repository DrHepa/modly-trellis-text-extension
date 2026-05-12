# modly-trellis-text-extension

Text-only TRELLIS extension for Modly.

Author: **DrHepa**

This repository is an additional Modly extension, not a replacement for the full
TRELLIS.2 extension. It exposes one capability only:

```text
prompt -> microsoft/TRELLIS-text-xlarge -> textured GLB mesh
```

## What this extension contains

- One Modly node: `trellis-text/text-to-mesh`
- Generator class: `TrellisTextGenerator`
- Native official TRELLIS text pipeline: `TrellisTextTo3DPipeline`
- GLB export through `trellis.utils.postprocessing_utils.to_glb`
- Auxiliary weight localization for text pipeline model references
- A vendoring blueprint for official `trellis/` and official `utils3d`

## Supported platforms

CUDA/NVIDIA is required. This is not a CPU or macOS-oriented extension.

Supported targets:

- Linux x86_64 + NVIDIA CUDA
- Linux ARM64/aarch64 + NVIDIA CUDA
- Windows + NVIDIA CUDA

Unsupported:

- macOS
- CPU-only runtime
- Non-NVIDIA GPU runtimes

## Model repositories

Primary model snapshot:

- `microsoft/TRELLIS-text-xlarge`

The native TRELLIS text pipeline also references hidden/auxiliary model assets,
including:

- `openai/clip-vit-large-patch14`

At runtime, `generator.py` rewrites pipeline model references into a localized
config when auxiliary checkpoint files are resolved from Hugging Face. First use
therefore still depends on valid Hugging Face access and cache/network state.

## What was intentionally removed

Compared to the full TRELLIS.2 extension, this text-only repository intentionally
does **not** include:

- `vendor/trellis2/`
- TRELLIS.2 image-to-mesh node `generate`
- TRELLIS.2 texture node `texture-mesh`
- `o-voxel`
- `CuMesh`
- DINOv3/RMBG dependencies and docs
- optional `nvdiffrec`
- image/texture diagnostics and multi-capability routing

It still keeps the native dependencies required by official TRELLIS text GLB
postprocessing:

- `spconv`
- one attention backend (`flash-attn` or `xformers`, platform-dependent)
- `nvdiffrast`
- `diff_gaussian_rasterization`
- `xatlas`, `pyvista`, `pymeshfix`, `igraph`

## Setup

Modly invokes `setup.py` with extension metadata. For local dry-run planning:

```bash
python3 setup.py --dry-run-plan 86 124
```

Real setup creates `venv/`, installs PyTorch CUDA wheels, Python runtime deps,
`spconv`, attention backend, `nvdiffrast`, and `diff_gaussian_rasterization`:

```bash
python3 setup.py /path/to/python /path/to/modly-trellis-text-extension 86 124
```

Platform notes:

- Linux ARM64 source-builds `cumm`/`spconv` and uses `flash-attn==2.7.3`.
- Windows uses `xformers` for attention.
- Linux x86_64 tries `xformers` first and falls back to `flash-attn==2.7.3`.

## Vendoring

This repository starts with only `vendor/.gitkeep`. Populate vendor sources with:

```bash
python3 build_vendor.py
```

`build_vendor.py` vendors only pure-Python sources:

- official Microsoft `TRELLIS` `trellis/` runtime package slices
- official TRELLIS `utils3d` fork
- small helper packages if needed

Native CUDA packages are deliberately installed into the extension venv by
`setup.py`; they must not be copied into `vendor/`.

## Static validation

Lightweight checks only:

```bash
python3 -m py_compile generator.py setup.py build_vendor.py validate_text_only_setup.py
python3 validate_text_only_setup.py
git status --short
```

Do not run builds, installs, model downloads, or vendor downloads for static
validation.
