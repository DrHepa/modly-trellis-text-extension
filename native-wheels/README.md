# Native Wheels

This directory contains the tooling to build and smoke-test native wheels for the TRELLIS text-only extension.

## Purpose

The goal is wheel-first installation for native TRELLIS postprocessing dependencies:

- `nvdiffrast`
- `diff_gaussian_rasterization`

If these wheels are published in the extension GitHub Releases, supported end-user installs can avoid local CUDA compilation. If wheels are missing or incompatible, `setup.py` falls back to the existing source-build path with a clear message.

## Supported wheel matrix

Phase 1 matrix for this tooling:

| Python ABI | Platform | Torch | TorchVision | CUDA tag |
| --- | --- | --- | --- | --- |
| `cp311` | `win_amd64` | `2.7.0+cu128` | `0.22.0+cu128` | `cu128` |
| `cp312` | `win_amd64` | `2.7.0+cu128` | `0.22.0+cu128` | `cu128` |
| `cp311` | `linux_x86_64` | `2.7.0+cu128` | `0.22.0+cu128` | `cu128` |
| `cp312` | `linux_x86_64` | `2.7.0+cu128` | `0.22.0+cu128` | `cu128` |

Expected release tags:

- `native-wheels-torch270-cu128-v2`
- `native-wheels-linux-x86_64-torch270-cu128-v1`

Compiled CUDA architecture targets:

- `sm_61`, `sm_75`, `sm_80`, `sm_86`, `sm_89`, plus `sm_90+PTX`

The Windows v2 release expands architecture coverage after v1 exposed `cudaErrorNoKernelImageForDevice` / CUDA error 209 on some Windows NVIDIA GPUs when `nvdiffrast` initialized its rasterizer kernels.

Linux phase 1 intentionally publishes hosted-Ubuntu `linux_x86_64` wheels, not manylinux wheels. WHY? Because the current target is a pragmatic first release built on GitHub-hosted `ubuntu-22.04` with a Conda CUDA toolchain. That is enough to prove the release flow and enable wheel-first installs on matching systems, but it is NOT a general manylinux portability promise.

Runtime validation so far:

- `text-to-mesh-base` has generated a textured mesh successfully on a Windows system with 8 GB VRAM using the v2 wheels.
- `text-to-mesh-large` has also generated a textured mesh successfully on a Windows system with 8 GB VRAM using the v2 wheels.
- `text-to-mesh` / XL still has the highest VRAM pressure; use Base first on constrained systems and Large when quality/headroom permits.

GitHub-hosted Linux CI has NO NVIDIA GPU, so the Linux workflow only performs import smoke checks. That proves packaging/import compatibility, not CUDA kernel execution. Manual smoke on a real Linux x86_64 NVIDIA GPU is still required before treating a Linux release as fully validated.

Expected GitHub repository:

- `https://github.com/DrHepa/modly-trellis-text-extension`

## Build requirements

- Windows x64 or Linux x86_64
- NVIDIA CUDA Toolkit 12.8
- Python 3.11 or 3.12
- Git
- PowerShell for Windows scripts
- Bash for Linux scripts

Windows additionally requires:

- Visual Studio Build Tools 2022 with Desktop development with C++

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

### Windows v2

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

### Linux x86_64 v1

Build `nvdiffrast`:

```bash
bash native-wheels/scripts/build-nvdiffrast-linux.sh --python python3 --out-dir native-wheels/wheelhouse/cp311
```

Build `diff_gaussian_rasterization`:

```bash
bash native-wheels/scripts/build-diff-gaussian-linux.sh --python python3 --out-dir native-wheels/wheelhouse/cp311
```

Useful parameters shared by both Linux scripts:

- `--python`
- `--out-dir`
- `--work-dir`
- `--torch-version`
- `--torchvision-version`
- `--torch-index-url`
- `--cuda-root`
- `--torch-cuda-arch-list`
- `--clean`

The Linux scripts clone the same pinned upstream refs as `setup.py`, export `CUDA_HOME`, `CUDA_PATH`, `CUDACXX`, `TORCH_CUDA_ARCH_LIST`, `PATH`, `CPATH`, `LIBRARY_PATH`, and `LD_LIBRARY_PATH`, then build with `python -m pip wheel --no-build-isolation`.

## Uploading GitHub Release assets

1. Create or update the appropriate release tag.
2. Upload the generated `.whl` files as release assets.
3. Keep filenames unchanged.
4. Verify the assets match the ABI/platform expected by `setup.py`:
   `nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl`
   `nvdiffrast-0.4.0-cp312-cp312-win_amd64.whl`
   `diff_gaussian_rasterization-0.0.0-cp311-cp311-win_amd64.whl`
   `diff_gaussian_rasterization-0.0.0-cp312-cp312-win_amd64.whl`
   `nvdiffrast-0.4.0-cp311-cp311-linux_x86_64.whl`
   `nvdiffrast-0.4.0-cp312-cp312-linux_x86_64.whl`
   `diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl`
   `diff_gaussian_rasterization-0.0.0-cp312-cp312-linux_x86_64.whl`

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

Linux smoke test command:

```bash
bash native-wheels/scripts/smoke-test-linux.sh --python python3 --wheel-dir native-wheels/wheelhouse/cp311
```

Important: the Linux smoke test is import-only because GitHub-hosted runners do not expose NVIDIA GPUs.

## GitHub Actions build flow

The repository also provides manual workflows:

```text
.github/workflows/build-native-windows-wheels.yml
.github/workflows/build-native-linux-wheels.yml
```

Run it from GitHub Actions or with `gh` after the workflow has been committed and pushed:

```powershell
gh workflow run build-native-windows-wheels.yml -f release_tag=native-wheels-torch270-cu128-v2 -f upload_release=true
```

The workflow builds `cp311` and `cp312` wheels on `windows-2022`, installs CUDA Toolkit 12.8 build components plus CUDA dev libraries (`cuBLAS`, `cuSPARSE`, `cuSOLVER`, `cuRAND`, `cuFFT`, and `cuda-cccl_win-64`) into a temporary Conda prefix from the NVIDIA Conda channel, runs the smoke test, and optionally uploads the wheels plus license files to the release tag.

The Linux workflow builds `cp311` and `cp312` wheels on `ubuntu-22.04`, installs CUDA Toolkit 12.8 build components plus CUDA dev libraries (`cuBLAS`, `cuSPARSE`, `cuSOLVER`, `cuRAND`, and `cuFFT`) into an explicit Conda prefix from the NVIDIA Conda channel, runs the Linux import-only smoke test, uploads artifacts named `native-linux-x86_64-wheels-cp311` and `native-linux-x86_64-wheels-cp312`, and optionally publishes the wheels plus license files to the Linux release tag.
