# Native Windows Wheels

This directory contains the initial tooling to build and smoke-test Windows-native wheels for the TRELLIS text-only extension.

## Purpose

The goal is wheel-first Windows installation for native TRELLIS postprocessing dependencies:

- `nvdiffrast`
- `diff_gaussian_rasterization`

If these wheels are published in the extension GitHub Releases, end-user Windows installs can avoid local CUDA/MSVC compilation. If wheels are missing or incompatible, `setup.py` falls back to the existing source-build path with a clear message.

## Supported wheel matrix

Minimum supported matrix for this tooling:

| Python ABI | Platform | Torch | TorchVision | CUDA tag |
| --- | --- | --- | --- | --- |
| `cp311` | `win_amd64` | `2.7.0+cu128` | `0.22.0+cu128` | `cu128` |
| `cp312` | `win_amd64` | `2.7.0+cu128` | `0.22.0+cu128` | `cu128` |

Expected release tag:

- `native-wheels-torch270-cu128-v1`

Expected GitHub repository:

- `https://github.com/DrHepa/modly-trellis-text-extension`

## Build requirements

- Windows x64
- NVIDIA CUDA Toolkit 12.8
- Visual Studio Build Tools 2022 with Desktop development with C++
- Python 3.11 or 3.12
- Git
- PowerShell

The native builds still depend on a functional CUDA compiler toolchain:

- `nvcc.exe`
- `cl.exe`
- Windows SDK headers/libs

## Legal warning

These upstream native dependencies are NOT general commercial redistributables:

- `nvdiffrast`: NVIDIA Source Code License, non-commercial / research / evaluation constraints
- `diff-gaussian-rasterization`: Gaussian Splatting / Inria research license, non-commercial / research / evaluation constraints

Read the full texts under `native-wheels/licenses/` before publishing release assets.

## Manual build flow

Build `nvdiffrast`:

```powershell
pwsh .\native-wheels\scripts\build-nvdiffrast.ps1 -Python py -OutDir .\native-wheels\wheelhouse\cp311
```

Build `diff_gaussian_rasterization`:

```powershell
pwsh .\native-wheels\scripts\build-diff-gaussian.ps1 -Python py -OutDir .\native-wheels\wheelhouse\cp311
```

Useful parameters shared by both scripts:

- `-Python`
- `-OutDir`
- `-WorkDir`
- `-TorchVersion`
- `-TorchVisionVersion`
- `-TorchIndexUrl`
- `-CudaRoot`
- `-TorchCudaArchList`
- `-Clean`

The scripts:

- install pinned `torch`/`torchvision` CUDA 12.8 wheels;
- prepare a local build venv;
- clone pinned upstream sources;
- build wheels with `python -m pip wheel --no-build-isolation -w <outdir>`;
- leave output in `wheelhouse/` for manual review and upload.

## Uploading GitHub Release assets

1. Create or update the release tag `native-wheels-torch270-cu128-v1`.
2. Upload the generated `.whl` files as release assets.
3. Keep filenames unchanged.
4. Verify the assets match the ABI/platform expected by `setup.py`:
   `nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl`
   `nvdiffrast-0.4.0-cp312-cp312-win_amd64.whl`
   `diff_gaussian_rasterization-0.0.0-cp311-cp311-win_amd64.whl`
   `diff_gaussian_rasterization-0.0.0-cp312-cp312-win_amd64.whl`

Important: `diff_gaussian_rasterization` upstream does not declare an explicit package version in its `setup.py`, so the wheel filename is expected to carry the default `0.0.0` version unless upstream changes.

## Smoke test

After building wheels locally:

```powershell
pwsh .\native-wheels\scripts\smoke-test.ps1 -WheelDir .\native-wheels\wheelhouse\cp311 -Python py
```

The smoke test creates a temporary venv, installs pinned Torch CUDA wheels plus the local native wheels, and verifies these imports:

- `torch`
- `nvdiffrast.torch`
- `diff_gaussian_rasterization`

If the smoke test fails, do NOT publish the assets.

## GitHub Actions build flow

The repository also provides a manual workflow:

```text
.github/workflows/build-native-windows-wheels.yml
```

Run it from GitHub Actions or with `gh` after the workflow has been committed and pushed:

```powershell
gh workflow run build-native-windows-wheels.yml -f release_tag=native-wheels-torch270-cu128-v1 -f upload_release=true
```

The workflow builds `cp311` and `cp312` wheels on `windows-2022`, installs CUDA Toolkit 12.8 build components from the NVIDIA Conda channel, runs the smoke test, and optionally uploads the wheels plus license files to the release tag.
