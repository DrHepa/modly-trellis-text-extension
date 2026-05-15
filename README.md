# Modly TRELLIS Text Extension

Native text-to-3D extension for [Modly](https://github.com/DrHepa) using the official TRELLIS text pipeline.

This repository provides a focused text-only runtime:

```text
text prompt -> selected microsoft/TRELLIS-text-* model -> textured GLB mesh
```

This extension is intentionally separate from the full TRELLIS.2 image/texturing extension. It is designed for users who only need prompt-to-mesh generation and do not want the additional image-to-mesh and mesh-texturing dependency surface.

## Features

- Three Modly nodes backed by official TRELLIS text checkpoints:
  - `text-to-mesh-base` — Base / low VRAM
  - `text-to-mesh-large` — Large / balanced
  - `text-to-mesh` — XL / original high-quality node kept for workflow compatibility
- Generator class: `TrellisTextGenerator`
- Official native TRELLIS text pipeline: `TrellisTextTo3DPipeline`
- GLB export through `trellis.utils.postprocessing_utils.to_glb`
- Localized auxiliary model references for TRELLIS text pipeline assets
- Reduced dependency surface compared with the full TRELLIS.2 extension

### Upstream Modly workflow compatibility

The runtime is text-to-mesh, but upstream Modly currently executes `model` workflow nodes through the image generation path and calls `readFileBase64()` before invoking the extension. For compatibility, every TRELLIS text node is therefore declared as `image -> mesh` in `manifest.json`.

The image input is a placeholder and is ignored by `TrellisTextGenerator`; generation is controlled by the `Prompt` parameter. In workflows, connect any valid image input or select an image before running, then enter the actual text prompt in the node parameters.

## Platform support

This extension requires an NVIDIA CUDA runtime. CPU-only execution, macOS, and non-NVIDIA GPU backends are not supported.

| Platform | Status | Notes |
| --- | --- | --- |
| Linux x86_64 + NVIDIA CUDA | Supported | Uses PyTorch CUDA wheels, prebuilt `spconv-*` when available, and `xformers` with `flash-attn` fallback. |
| Linux ARM64/aarch64 + NVIDIA CUDA | Supported | Builds `cumm`/`spconv` from source and applies an ARM64 CUDA discovery hotfix before building `spconv`. |
| Windows + NVIDIA CUDA | Supported | Uses PyTorch CUDA wheels, `xformers`, prebuilt `spconv-*` packages, and wheel-first installation for TRELLIS native postprocessing extensions. Falls back to source builds when native release wheels are unavailable or incompatible. |
| macOS | Not supported | The TRELLIS text runtime depends on CUDA and calls `.cuda()`. |

## Models

Official model repositories exposed as separate nodes:

| Node | Hugging Face repo | Use case |
| --- | --- | --- |
| `text-to-mesh-base` | `microsoft/TRELLIS-text-base` | Lowest-VRAM official text model; recommended first choice for 8 GB Windows laptops. |
| `text-to-mesh-large` | `microsoft/TRELLIS-text-large` | Balanced quality/VRAM target. |
| `text-to-mesh` | `microsoft/TRELLIS-text-xlarge` | Original XL node; highest VRAM pressure and best kept for 16 GB+ GPUs. |

The model size is intentionally represented as separate nodes instead of a runtime parameter. Modly tracks downloads, readiness, and model directories per node/capability, while the official TRELLIS text variants live in separate Hugging Face repositories.

Additional model assets used by the native TRELLIS text pipeline include:

- `openai/clip-vit-large-patch14`

On first load, the generator rewrites TRELLIS pipeline references into a localized configuration when auxiliary checkpoint files are resolved from Hugging Face. A working Hugging Face cache or network access is therefore required for the first successful run.

## What is intentionally excluded

This repository does not include the full TRELLIS.2 image/texturing stack:

- no `vendor/trellis2/`
- no image-to-mesh `generate` node
- no `texture-mesh` node
- no `o-voxel`
- no `CuMesh`
- no DINOv3/RMBG image dependencies
- no optional `nvdiffrec`
- no multi-capability routing for image/texturing workflows

The text runtime still requires several CUDA/native packages because official TRELLIS GLB postprocessing uses mesh and Gaussian rendering components:

- `spconv`
- `xformers` or `flash-attn`, depending on platform
- `nvdiffrast`
- `diff_gaussian_rasterization`
- `xatlas`, `pyvista`, `pymeshfix`, `igraph`

## Installation flow

Modly invokes `setup.py` automatically when the extension is installed. The setup script creates an isolated `venv/` inside the extension directory and installs the CUDA/native runtime dependencies there.

All pip operations are executed as `python -m pip` inside the extension venv. This is intentional: on Windows, directly executing `venv\\Scripts\\pip.exe` while upgrading `pip` can fail because the wrapper is trying to replace itself.

During setup, `build_vendor.py` is also executed automatically if `vendor/` does not already contain the official TRELLIS text runtime sources. This is required for runtime imports such as `trellis.pipelines.TrellisTextTo3DPipeline`. If vendor population fails, check network access to GitHub and PyPI, then rerun extension setup.

The vendoring step patches TRELLIS pipeline exports for this text-only extension. Official TRELLIS exposes image and text pipelines from the same package; importing the image pipeline pulls image-only dependencies such as `rembg`. This extension rewrites `trellis/pipelines/__init__.py` so only `TrellisTextTo3DPipeline` is exported.

The vendoring step also makes `open3d` optional inside the official text pipeline. TRELLIS imports `open3d` for mesh-conditioned variant generation, but the Modly node only uses prompt-to-mesh generation. Requiring `open3d` at import time would break Linux ARM64 and other platforms where Open3D wheels are not available.

Setup uses a versioned marker, `vendor/.trellis-text-only-v4`, so existing installs with older vendored TRELLIS sources are regenerated automatically.

The vendoring step also patches TRELLIS `from_pretrained()` to accept a `config_file` argument. This is required because the extension localizes auxiliary model references into `pipeline.text-localized.json` before loading the native text pipeline.

The vendoring step also removes the hard `kaolin` dependency from the FlexiCubes helper. Upstream FlexiCubes imports `kaolin.utils.testing.check_tensor` only for shape assertions; this extension patches in an equivalent local helper instead of pulling the full Kaolin native stack.

For local install-plan diagnostics only:

```bash
python3 setup.py --dry-run-plan 86 124
```

For a direct local setup invocation:

```bash
python3 setup.py /path/to/python /path/to/modly-trellis-text-extension 86 124
```

Arguments:

- `python_exe`: Python executable used to create the extension venv.
- `ext_dir`: extension installation directory.
- `gpu_sm`: GPU SM value reported by Modly.
- `cuda_version`: CUDA version reported by Modly, for example `124` or `128`.

### Linux ARM64 note

On Linux ARM64, setup builds `cumm` and `spconv` from source. CUDA toolkit discovery is explicitly steered through `CUDA_HOME`, `CUDA_PATH`, and `CUDACXX`. After `cumm` is installed, setup patches `cumm/common.py` so the subsequent `spconv` build uses the selected CUDA toolkit root instead of accidentally falling back to `/usr/local/cuda`.

This avoids mixed-toolkit failures such as:

```text
macro "__cudaLaunch" requires 2 arguments, but only 1 given
```

which can happen when `nvcc` comes from `/usr/local/cuda-12.8` but headers are discovered from a different `/usr/local/cuda` installation.

### Windows native build note

Windows installation compiles native CUDA extensions such as `diff_gaussian_rasterization` and `nvdiffrast`. The setup script prepares the native build environment by:

- resolving the NVIDIA CUDA Toolkit through `CUDA_HOME`, `CUDA_PATH`, `MODLY_TRELLIS_TEXT_CUDA_TOOLKIT_ROOT`, or the default CUDA Toolkit install directory;
- locating Visual Studio Build Tools through `vswhere.exe`;
- loading `vcvars64.bat` so `cl.exe`, `INCLUDE`, `LIB`, and `PATH` are available to PyTorch CUDA extension builds;
- setting `DISTUTILS_USE_SDK=1` for setuptools/PyTorch native extension builds.

If Windows setup fails while compiling a native extension, verify that **Visual Studio Build Tools 2022** is installed with the **Desktop development with C++** workload, including MSVC v143 and a Windows SDK. Also verify that a matching NVIDIA CUDA Toolkit is installed.

For current PyTorch `cu128` installs, this usually means installing the **CUDA Toolkit 12.8** from NVIDIA, not just the GPU driver. If CUDA is installed in a non-standard location, set `MODLY_TRELLIS_TEXT_CUDA_TOOLKIT_ROOT` to the Toolkit root before installing the extension.

Before source-building these TRELLIS-native extensions on Windows, setup now tries extension-owned GitHub Release wheels for `nvdiffrast` and `diff_gaussian_rasterization`. The default release channel targets `cp311`/`cp312` on `win_amd64` for `torch==2.7.0`, `torchvision==0.22.0`, and `cu128`. You can opt out with `MODLY_TRELLIS_TEXT_DISABLE_NATIVE_WHEELS=1` or override the release base URL with `MODLY_TRELLIS_TEXT_NATIVE_WHEEL_BASE_URL`.

The wheel build/upload helper docs live under `native-wheels/`, including the full upstream license texts required for redistribution review. This matters because both native packages carry non-commercial / research-oriented licensing constraints.

Windows setup is wheel-first for dependencies that upstream publishes as wheels. It installs PyTorch, runtime Python dependencies, `spconv` prebuilt wheels, and `xformers` before preparing the native compiler environment for source-built TRELLIS postprocessing extensions. The setup uses only known upstream `spconv` prebuilt CUDA wheel tags (`cu120`, `cu118`) instead of trying every PyTorch CUDA tag. This avoids misleading `spconv-cu128` errors on Windows while still allowing the prebuilt fallback path that upstream currently publishes.

`xformers` is pinned to the selected PyTorch version and installed without dependencies so pip cannot silently replace the `torch`/`torchvision` pair. Setup validates the Torch stack after installing PyTorch and again after installing the attention backend.

When verifying `spconv`, setup imports `torch` first so Windows registers PyTorch/CUDA DLL directories before `spconv` loads its native modules. This mirrors the runtime generator import order.

On Windows Python 3.12, setup only tries `spconv-cu118` because upstream currently publishes `cp312-win_amd64` wheels for `spconv-cu118` but not for `spconv-cu120`. Setup and runtime also suppress upstream `spconv` `FutureWarning` messages that can otherwise break smoke checks in environments that treat warnings as errors.

## Vendoring

The repository starts with a placeholder `vendor/.gitkeep`. Modly setup populates pure-Python vendor sources automatically. For development, you can also run the vendoring step manually:

```bash
python3 build_vendor.py
```

`build_vendor.py` vendors only pure-Python runtime sources:

- official Microsoft TRELLIS `trellis/` runtime slices
- official TRELLIS-compatible `utils3d` fork
- small pure-Python helper packages where useful

Native CUDA packages must be installed by `setup.py` into the extension venv. They must not be copied into `vendor/`.

## Static validation

Use static validation before publishing changes:

```bash
python3 -m py_compile generator.py setup.py build_vendor.py validate_text_only_setup.py
python3 validate_text_only_setup.py
git status --short
```

These checks do not install dependencies, build native packages, download models, or populate `vendor/`.

## Repository status

This is a text-only Modly extension. The full TRELLIS.2 extension should remain available separately for image-to-mesh and mesh-texturing workflows.
