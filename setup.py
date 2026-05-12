"""
Setup for the Modly TRELLIS text-only extension.

This script creates the extension venv and installs CUDA/native dependencies
needed by the official TRELLIS text pipeline. It intentionally excludes the
TRELLIS.2 image/texturing stack and related native packages.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


FLASH_ATTN_VERSION = "2.7.3"
SPCONV_SOURCE_REPO = "https://github.com/traveller59/spconv.git"
SPCONV_SOURCE_REF = "v2.3.8"
CUMM_SOURCE_REPO = "https://github.com/FindDefinition/cumm.git"
CUMM_SOURCE_REF = "v0.7.11"
NVDIFFRAST_SOURCE_REPO = "https://github.com/NVlabs/nvdiffrast.git"
NVDIFFRAST_SOURCE_REF = "v0.4.0"
MIP_SPLATTING_SOURCE_REPO = "https://github.com/autonomousvision/mip-splatting.git"
MIP_SPLATTING_SOURCE_REF = "dda02ab5ecf45d6edb8c540d9bb65c7e451345a9"
MIP_SPLATTING_DIFF_GAUSSIAN_SUBDIRECTORY = "submodules/diff-gaussian-rasterization"

PYTHON_RUNTIME_DEPENDENCIES = (
    "Pillow",
    "numpy",
    "opencv-python-headless",
    "huggingface_hub",
    "transformers>=4.46.0",
    "accelerate",
    "safetensors",
    "imageio",
    "imageio-ffmpeg",
    "easydict",
    "tqdm",
    "trimesh",
    "scipy",
    "scikit-image",
    "kornia",
    "timm",
    "ninja",
    "xatlas",
    "pyvista",
    "pymeshfix",
    "igraph",
)

CUMM_CUDA_DISCOVERY_PATCH_MARKER = "modly_trellis_text_cuda_root_override"
CUMM_SUPPORTED_CUDA_ARCHES = frozenset({"5.2", "6.0", "6.1", "7.0", "7.2", "7.5", "8.0", "8.6", "8.7", "8.9", "9.0"})
CUMM_MAX_SUPPORTED_SM = 90
CUMM_MAX_SUPPORTED_ARCH = "9.0"
CUMM_FORWARD_COMPAT_ARCH = "9.0+PTX"


@dataclass(frozen=True)
class PlatformInstallPlan:
    name: str
    attention_backends: tuple[tuple[str, str], ...]


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_linux() -> bool:
    return platform.system() == "Linux"


def machine_arch() -> str:
    return platform.machine().lower()


def platform_label() -> str:
    return f"{platform.system()} {machine_arch()}"


def is_linux_arm64() -> bool:
    return is_linux() and machine_arch() in {"aarch64", "arm64"}


def cuda_arch_string_from_sm(gpu_sm: int) -> str | None:
    if gpu_sm <= 0:
        return None
    major, minor = divmod(gpu_sm, 10)
    return f"{major}.{minor}"


def resolve_cumm_cuda_arch(gpu_sm: int) -> tuple[str | None, str]:
    requested_arch = cuda_arch_string_from_sm(gpu_sm)
    if requested_arch is None:
        return None, "gpu_sm was not provided; upstream CUDA arch autodetection will be used"
    if requested_arch in CUMM_SUPPORTED_CUDA_ARCHES:
        return requested_arch, f"SM {gpu_sm} maps directly to supported cumm arch {requested_arch}"
    if gpu_sm > CUMM_MAX_SUPPORTED_SM:
        return CUMM_FORWARD_COMPAT_ARCH, f"SM {gpu_sm} maps to unsupported arch {requested_arch}; clamping to {CUMM_FORWARD_COMPAT_ARCH} because cumm {CUMM_SOURCE_REF} supports up to {CUMM_MAX_SUPPORTED_ARCH}"
    return requested_arch, f"SM {gpu_sm} maps to arch {requested_arch}; no compatibility remap applied"


def plan_platform_install() -> PlatformInstallPlan:
    if is_linux_arm64():
        return PlatformInstallPlan(name="linux-arm64", attention_backends=(("flash_attn", f"flash-attn=={FLASH_ATTN_VERSION}"),))
    if is_windows():
        return PlatformInstallPlan(name=f"windows-{machine_arch()}", attention_backends=(("xformers", "xformers"),))
    return PlatformInstallPlan(name=f"linux-{machine_arch()}", attention_backends=(("xformers", "xformers"), ("flash_attn", f"flash-attn=={FLASH_ATTN_VERSION}")))


def select_torch(gpu_sm: int, cuda_version: int) -> tuple[list[str], str, str]:
    if gpu_sm >= 100 or cuda_version >= 128:
        return ["torch==2.7.0", "torchvision==0.22.0"], "https://download.pytorch.org/whl/cu128", "cu128"
    if gpu_sm == 0 or gpu_sm >= 70:
        return ["torch==2.6.0", "torchvision==0.21.0"], "https://download.pytorch.org/whl/cu124", "cu124"
    return ["torch==2.5.1", "torchvision==0.20.1"], "https://download.pytorch.org/whl/cu118", "cu118"


def venv_bin(venv: Path, name: str) -> Path:
    if is_windows():
        suffix = ".exe" if not name.endswith(".exe") else ""
        return venv / "Scripts" / f"{name}{suffix}"
    return venv / "bin" / name


def prepend_directory_to_path(env: dict[str, str], directory: Path) -> dict[str, str]:
    updated = env.copy()
    existing = [part for part in updated.get("PATH", "").split(os.pathsep) if part]
    updated["PATH"] = os.pathsep.join([str(directory), *[part for part in existing if part != str(directory)]])
    return updated


def prepend_env_path(env: dict[str, str], key: str, *entries: Path) -> None:
    values = [str(entry) for entry in entries if str(entry)]
    existing = [part for part in env.get(key, "").split(os.pathsep) if part]
    env[key] = os.pathsep.join([*values, *[part for part in existing if part not in values]])


def cuda_version_to_toolkit_version(cuda_version: int) -> str | None:
    if cuda_version <= 0:
        return None
    major, minor = divmod(cuda_version, 10)
    return f"{major}.{minor}"


def candidate_cuda_toolkit_roots(cuda_version: int, env: dict[str, str] | None = None) -> list[Path]:
    source_env = env or os.environ
    candidates: list[Path] = []
    for key in ("MODLY_TRELLIS_TEXT_CUDA_TOOLKIT_ROOT", "CUDA_HOME", "CUDA_PATH"):
        raw = source_env.get(key)
        if raw:
            candidates.append(Path(raw).expanduser())
    toolkit_version = cuda_version_to_toolkit_version(cuda_version)
    if toolkit_version:
        candidates.append(Path(f"/usr/local/cuda-{toolkit_version}"))
    candidates.append(Path("/usr/local/cuda"))
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate)
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(candidate)
    return deduped


def resolve_cuda_toolkit_root(cuda_version: int, env: dict[str, str] | None = None) -> Path | None:
    for candidate in candidate_cuda_toolkit_roots(cuda_version, env=env):
        if candidate.exists():
            return candidate
    return None


def cuda_toolkit_library_dirs(toolkit_root: Path) -> tuple[Path, ...]:
    candidates = [toolkit_root / "lib64"]
    if is_linux_arm64():
        candidates.extend([toolkit_root / "targets" / "aarch64-linux" / "lib", toolkit_root / "targets" / "sbsa-linux" / "lib"])
    elif is_linux():
        candidates.append(toolkit_root / "targets" / "x86_64-linux" / "lib")
    return tuple(path for path in candidates if path.exists())


def source_build_env_overrides(*, gpu_sm: int, cuda_version: int, build_env: dict[str, str] | None = None, venv: Path | None = None) -> tuple[dict[str, str], dict[str, object]]:
    source_env = dict(build_env or os.environ)
    source_env.setdefault("CUMM_DISABLE_JIT", "1")
    source_env.setdefault("SPCONV_DISABLE_JIT", "1")
    diagnostics: dict[str, object] = {"CUMM_DISABLE_JIT": "1", "SPCONV_DISABLE_JIT": "1"}
    requested, reason = resolve_cumm_cuda_arch(gpu_sm)
    diagnostics["cumm_cuda_arch"] = {"requested": cuda_arch_string_from_sm(gpu_sm), "resolved": requested, "reason": reason}
    if requested:
        source_env.setdefault("CUMM_CUDA_ARCH_LIST", requested)
        diagnostics["CUMM_CUDA_ARCH_LIST"] = source_env["CUMM_CUDA_ARCH_LIST"]
    venv_bin_dir = venv_bin(venv, "python").parent if venv is not None else None
    if venv_bin_dir is not None:
        source_env = prepend_directory_to_path(source_env, venv_bin_dir)
    toolkit_root = resolve_cuda_toolkit_root(cuda_version, env=source_env)
    diagnostics["cuda_toolkit_root_candidates"] = [str(path) for path in candidate_cuda_toolkit_roots(cuda_version, env=source_env)]
    if toolkit_root is None:
        diagnostics["cuda_toolkit_root"] = None
        return source_env, diagnostics
    source_env["CUDA_HOME"] = str(toolkit_root)
    source_env["CUDA_PATH"] = str(toolkit_root)
    source_env["CUDACXX"] = str(toolkit_root / "bin" / "nvcc")
    prepend_env_path(source_env, "PATH", *(entry for entry in (venv_bin_dir, toolkit_root / "bin") if entry is not None))
    include_dir = toolkit_root / "include"
    prepend_env_path(source_env, "CPATH", include_dir)
    prepend_env_path(source_env, "C_INCLUDE_PATH", include_dir)
    prepend_env_path(source_env, "CPLUS_INCLUDE_PATH", include_dir)
    library_dirs = cuda_toolkit_library_dirs(toolkit_root)
    if library_dirs:
        prepend_env_path(source_env, "LIBRARY_PATH", *library_dirs)
        prepend_env_path(source_env, "LD_LIBRARY_PATH", *library_dirs)
    diagnostics["cuda_toolkit_root"] = str(toolkit_root)
    diagnostics["CUDA_HOME"] = source_env["CUDA_HOME"]
    diagnostics["CUDACXX"] = source_env["CUDACXX"]
    return source_env, diagnostics


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print("[setup] $", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd else None)


def pip(venv: Path, *args: str, env: dict[str, str] | None = None) -> None:
    run([str(venv_bin(venv, "pip")), *args], env=env)


def pip_install(venv: Path, *packages: str, env: dict[str, str] | None = None, no_build_isolation: bool = False) -> None:
    cmd = ["install"]
    if no_build_isolation:
        cmd.append("--no-build-isolation")
    cmd.extend(packages)
    pip(venv, *cmd, env=env)


def python(venv: Path, *args: str, env: dict[str, str] | None = None) -> None:
    run([str(venv_bin(venv, "python")), *args], env=env)


def native_install_error(package_name: str, attempted_ref: str, exc: Exception) -> RuntimeError:
    return RuntimeError(f"Failed to install native dependency '{package_name}' on {platform_label()} from {attempted_ref}. Cause: {exc}")


def clone_repo(dest: Path, repo: str, *, ref: str | None = None, recursive: bool = False) -> Path:
    run(["git", "clone", repo, str(dest)])
    if ref:
        run(["git", "checkout", ref], cwd=dest)
    if recursive:
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=dest)
    return dest


def install_from_repo(venv: Path, tmpdir: Path, folder_name: str, repo: str, *, ref: str, recursive: bool = False, subdirectory: str | None = None, env: dict[str, str] | None = None, no_deps: bool = False) -> None:
    try:
        checkout = clone_repo(tmpdir / folder_name, repo, ref=ref, recursive=recursive)
        package_dir = checkout / subdirectory if subdirectory else checkout
        cmd = ["install", "--no-build-isolation"]
        if no_deps:
            cmd.append("--no-deps")
        cmd.append(str(package_dir))
        pip(venv, *cmd, env=env)
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        raise native_install_error(folder_name, ref, exc) from exc


def install_packages_with_diagnostics(venv: Path, package_name: str, attempted_ref: str, *packages: str, env: dict[str, str] | None = None, no_build_isolation: bool = False) -> None:
    try:
        pip_install(venv, *packages, env=env, no_build_isolation=no_build_isolation)
    except subprocess.CalledProcessError as exc:
        raise native_install_error(package_name, attempted_ref, exc) from exc


def attention_backend_needs_no_build_isolation(backend_name: str, requirement: str) -> bool:
    return is_linux_arm64() and backend_name == "flash_attn" and requirement == f"flash-attn=={FLASH_ATTN_VERSION}"


def uninstall_packages(venv: Path, *packages: str) -> None:
    if packages:
        pip(venv, "uninstall", "-y", *packages)


def smoke_check_spconv(venv: Path, *, env: dict[str, str] | None = None) -> None:
    python(venv, "-c", "import spconv.pytorch as spconv; print('[setup] spconv import OK:', getattr(spconv, '__version__', 'unknown'))", env=env)


def install_prebuilt_spconv(venv: Path, cuda_tag: str) -> None:
    fallbacks = [cuda_tag, "cu128", "cu124", "cu122", "cu121", "cu120", "cu118"]
    tried: list[str] = []
    last_error: subprocess.CalledProcessError | None = None
    for tag in fallbacks:
        pkg = f"spconv-{tag}"
        if pkg in tried:
            continue
        tried.append(pkg)
        try:
            pip(venv, "install", pkg)
            smoke_check_spconv(venv)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"[setup] {pkg} not available; trying next fallback.")
    raise RuntimeError(f"Failed to install spconv. Tried: {', '.join(tried)}") from last_error


def install_spconv_from_source(venv: Path, gpu_sm: int, cuda_version: int, build_env: dict[str, str]) -> None:
    source_env, diagnostics = source_build_env_overrides(gpu_sm=gpu_sm, cuda_version=cuda_version, build_env=build_env, venv=venv)
    print(f"[setup] Linux ARM64 source build for cumm/spconv: {json.dumps(diagnostics, indent=2)}")
    uninstall_packages(venv, "spconv", "cumm")
    install_packages_with_diagnostics(venv, "spconv-build-prereqs", "pccm/ccimport/pybind11/fire", "pccm>=0.4.16", "ccimport>=0.4.4", "pybind11>=2.6.0", "fire", env=source_env)
    with tempfile.TemporaryDirectory(prefix="trellis-text-spconv-") as tmp:
        tmpdir = Path(tmp)
        install_from_repo(venv, tmpdir, "cumm", CUMM_SOURCE_REPO, ref=CUMM_SOURCE_REF, env=source_env, no_deps=True)
        install_from_repo(venv, tmpdir, "spconv", SPCONV_SOURCE_REPO, ref=SPCONV_SOURCE_REF, env=source_env, no_deps=True)
    smoke_check_spconv(venv, env=source_env)


def install_spconv(venv: Path, cuda_tag: str, gpu_sm: int, build_env: dict[str, str]) -> None:
    if is_linux_arm64():
        cuda_version = int(cuda_tag[2:]) if cuda_tag.startswith("cu") else 0
        install_spconv_from_source(venv, gpu_sm, cuda_version, build_env)
    else:
        install_prebuilt_spconv(venv, cuda_tag)


def install_attention_backend(venv: Path, plan: PlatformInstallPlan) -> str:
    failures: list[str] = []
    for backend_name, requirement in plan.attention_backends:
        try:
            pip_install(venv, requirement, no_build_isolation=attention_backend_needs_no_build_isolation(backend_name, requirement))
            return backend_name
        except subprocess.CalledProcessError as exc:
            failures.append(str(native_install_error(backend_name, requirement, exc)))
            print(f"[setup] {backend_name} install failed; trying next supported backend.")
    raise RuntimeError("No supported sparse attention backend could be installed.\n" + "\n\n".join(failures))


def install_core_native_dependencies(venv: Path, tmpdir: Path, build_env: dict[str, str]) -> None:
    install_from_repo(venv, tmpdir, "diff_gaussian_rasterization", MIP_SPLATTING_SOURCE_REPO, ref=MIP_SPLATTING_SOURCE_REF, recursive=True, subdirectory=MIP_SPLATTING_DIFF_GAUSSIAN_SUBDIRECTORY, env=build_env)
    install_from_repo(venv, tmpdir, "nvdiffrast", NVDIFFRAST_SOURCE_REPO, ref=NVDIFFRAST_SOURCE_REF, env=build_env)


def install_python_runtime_dependencies(venv: Path) -> None:
    pip(venv, "install", *PYTHON_RUNTIME_DEPENDENCIES)


def describe_install_plan(gpu_sm: int, cuda_version: int) -> dict[str, object]:
    torch_pkgs, torch_index, cuda_tag = select_torch(gpu_sm, cuda_version)
    plan = plan_platform_install()
    description: dict[str, object] = {
        "platform": platform_label(),
        "plan": plan.name,
        "torch_packages": torch_pkgs,
        "torch_index": torch_index,
        "cuda_tag": cuda_tag,
        "spconv_strategy": "source" if is_linux_arm64() else "prebuilt",
        "attention_backends": [backend for backend, _ in plan.attention_backends],
        "native_from_git": {
            "nvdiffrast": f"{NVDIFFRAST_SOURCE_REPO}@{NVDIFFRAST_SOURCE_REF}",
            "diff_gaussian_rasterization": f"{MIP_SPLATTING_SOURCE_REPO}@{MIP_SPLATTING_SOURCE_REF}:{MIP_SPLATTING_DIFF_GAUSSIAN_SUBDIRECTORY}",
        },
        "excluded": ["TRELLIS.2 image/texturing", "o-voxel", "CuMesh", "DINOv3", "RMBG", "nvdiffrec"],
    }
    if is_linux_arm64():
        _, diagnostics = source_build_env_overrides(gpu_sm=gpu_sm, cuda_version=cuda_version)
        description["source_build_env"] = diagnostics
    return description


def setup(python_exe: str, ext_dir: Path, gpu_sm: int, cuda_version: int = 0) -> None:
    venv = ext_dir / "venv"
    build_env = os.environ.copy()
    build_env.setdefault("CUDAFLAGS", "-allow-unsupported-compiler")
    build_env.setdefault("CMAKE_CUDA_FLAGS", "-allow-unsupported-compiler")
    plan = plan_platform_install()

    print(f"[setup] Platform install plan: {plan.name} ({platform_label()})")
    run([python_exe, "-m", "venv", str(venv)])
    pip(venv, "install", "--upgrade", "pip", "setuptools", "wheel")

    torch_pkgs, torch_index, cuda_tag = select_torch(gpu_sm, cuda_version)
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)
    install_python_runtime_dependencies(venv)
    install_spconv(venv, cuda_tag, gpu_sm, build_env)
    chosen_attention_backend = install_attention_backend(venv, plan)
    print(f"[setup] Selected sparse attention backend: {chosen_attention_backend}")

    native_build_env, native_diagnostics = source_build_env_overrides(gpu_sm=gpu_sm, cuda_version=cuda_version, build_env=build_env, venv=venv)
    if native_diagnostics.get("cuda_toolkit_root"):
        print(f"[setup] Steering native source builds to CUDA toolkit root: {native_diagnostics['cuda_toolkit_root']}")
    with tempfile.TemporaryDirectory(prefix="trellis-text-setup-") as tmp:
        install_core_native_dependencies(venv, Path(tmp), native_build_env)

    print("[setup] Done. Extension venv is ready at:", venv)
    print("[setup] First runtime load still requires Hugging Face access for microsoft/TRELLIS-text-xlarge and hidden CLIP assets.")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--dry-run-plan":
        gpu_sm = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
        cuda_version = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
        print(json.dumps(describe_install_plan(gpu_sm, cuda_version), indent=2))
    elif len(sys.argv) >= 4:
        setup(sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) >= 5 else 0)
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(args["python_exe"], Path(args["ext_dir"]), int(args.get("gpu_sm", 0)), int(args.get("cuda_version", 0)))
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":124}\'')
        print("   or: python setup.py --dry-run-plan [gpu_sm] [cuda_version]")
        sys.exit(1)
