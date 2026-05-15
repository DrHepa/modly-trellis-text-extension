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
import shutil
import subprocess
import sys
import sysconfig
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
NATIVE_WHEEL_RELEASE_REPO = "DrHepa/modly-trellis-text-extension"
NATIVE_WHEEL_RELEASE_TAG = "native-wheels-torch270-cu128-v1"
NATIVE_WHEEL_SUPPORTED_CUDA_TAG = "cu128"
NATIVE_WHEEL_SUPPORTED_TORCH = "2.7.0"
NATIVE_WHEEL_SUPPORTED_TORCHVISION = "0.22.0"
NATIVE_WHEEL_FILENAMES = {
    "nvdiffrast": {
        "filename": "nvdiffrast-0.4.0-{abi}-{abi}-win_amd64.whl",
        "import": "nvdiffrast.torch",
    },
    "diff_gaussian_rasterization": {
        "filename": "diff_gaussian_rasterization-0.0.0-{abi}-{abi}-win_amd64.whl",
        "import": "diff_gaussian_rasterization",
    },
}
VENDOR_REQUIRED_PATHS = (
    Path("vendor") / "trellis" / "__init__.py",
    Path("vendor") / "trellis" / "pipelines" / "trellis_text_to_3d.py",
    Path("vendor") / "utils3d",
    Path("vendor") / ".trellis-text-only-v4",
)

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
    "plyfile",
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
KNOWN_PREBUILT_SPCONV_CUDA_TAGS = ("cu120", "cu118")
XFORMERS_BY_TORCH_VERSION = {
    "2.7.0": "xformers==0.0.30",
    "2.6.0": "xformers==0.0.29.post3",
    "2.5.1": "xformers==0.0.28.post3",
}


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


def python_abi_tag() -> str | None:
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return abi if abi in {"cp311", "cp312"} else None


def wheel_platform_tag() -> str:
    return sysconfig.get_platform().replace("-", "_").replace(".", "_")


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


def package_version(packages: list[str], name: str) -> str:
    prefix = f"{name}=="
    for package in packages:
        if package.startswith(prefix):
            return package[len(prefix):]
    raise RuntimeError(f"Internal setup error: missing pinned package '{name}' in {packages}")


def resolve_attention_backends(plan: PlatformInstallPlan, torch_packages: list[str]) -> tuple[tuple[str, str], ...]:
    torch_version = package_version(torch_packages, "torch")
    resolved: list[tuple[str, str]] = []
    for backend_name, requirement in plan.attention_backends:
        if backend_name == "xformers" and requirement == "xformers":
            try:
                requirement = XFORMERS_BY_TORCH_VERSION[torch_version]
            except KeyError as exc:
                raise RuntimeError(f"No pinned xformers version is known for torch=={torch_version}") from exc
        resolved.append((backend_name, requirement))
    return tuple(resolved)


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
    if is_windows():
        program_files = source_env.get("ProgramFiles", r"C:\Program Files")
        cuda_base = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if toolkit_version:
            candidates.append(cuda_base / f"v{toolkit_version}")
        candidates.extend(sorted(cuda_base.glob("v*"), reverse=True) if cuda_base.exists() else [])
    else:
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
    if is_windows():
        return tuple(path for path in (toolkit_root / "lib" / "x64", toolkit_root / "lib") if path.exists())

    candidates = [toolkit_root / "lib64"]
    if is_linux_arm64():
        candidates.extend([toolkit_root / "targets" / "aarch64-linux" / "lib", toolkit_root / "targets" / "sbsa-linux" / "lib"])
    elif is_linux():
        candidates.append(toolkit_root / "targets" / "x86_64-linux" / "lib")
    return tuple(path for path in candidates if path.exists())


def command_available(command: str, env: dict[str, str]) -> bool:
    checker = ["where", command] if is_windows() else ["which", command]
    return subprocess.run(checker, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env).returncode == 0


def candidate_vswhere_paths(env: dict[str, str]) -> list[Path]:
    candidates = []
    for key, fallback in (("ProgramFiles(x86)", r"C:\Program Files (x86)"), ("ProgramFiles", r"C:\Program Files")):
        root = env.get(key, fallback)
        candidates.append(Path(root) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    return candidates


def parse_windows_set_output(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            parsed[key] = value
    return parsed


def resolve_windows_msvc_env(build_env: dict[str, str]) -> tuple[dict[str, str], dict[str, object]]:
    env = dict(build_env)
    diagnostics: dict[str, object] = {"strategy": "existing-path"}

    if command_available("cl.exe", env):
        diagnostics["cl.exe"] = "found-on-path"
        env.setdefault("DISTUTILS_USE_SDK", "1")
        env.setdefault("MSSdk", "1")
        return env, diagnostics

    vswhere = next((path for path in candidate_vswhere_paths(env) if path.exists()), None)
    diagnostics["vswhere_candidates"] = [str(path) for path in candidate_vswhere_paths(env)]
    if vswhere is None:
        raise RuntimeError(
            "Windows native CUDA builds require Microsoft Visual Studio Build Tools 2022. "
            "Install the 'Desktop development with C++' workload, including MSVC v143 and a Windows SDK. "
            "Could not find vswhere.exe to locate the toolchain."
        )

    install_path = subprocess.check_output(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        text=True,
        env=env,
    ).strip()
    if not install_path:
        raise RuntimeError(
            "Windows native CUDA builds require MSVC. Visual Studio was found, but the VC++ x64 tools "
            "component is missing. Install Visual Studio Build Tools 2022 with 'Desktop development with C++'."
        )

    vcvars = Path(install_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.exists():
        raise RuntimeError(f"Could not find vcvars64.bat at {vcvars}. Repair Visual Studio Build Tools 2022.")

    vcvars_cmd = f'"{vcvars}" amd64 >nul && set'
    vcvars_output = subprocess.check_output(["cmd.exe", "/s", "/c", vcvars_cmd], text=True, env=env)
    msvc_env = parse_windows_set_output(vcvars_output)
    merged = dict(env)
    merged.update(msvc_env)
    merged.setdefault("DISTUTILS_USE_SDK", "1")
    merged.setdefault("MSSdk", "1")
    if not command_available("cl.exe", merged):
        raise RuntimeError("vcvars64.bat completed but cl.exe is still not available on PATH.")

    diagnostics.update({"strategy": "vswhere-vcvars64", "vswhere": str(vswhere), "vcvars64": str(vcvars), "cl.exe": "found-after-vcvars"})
    return merged, diagnostics


def resolve_native_build_env(
    venv: Path,
    *,
    gpu_sm: int,
    cuda_version: int,
    build_env: dict[str, str],
) -> tuple[dict[str, str], dict[str, object] | None]:
    if is_linux_arm64():
        return source_build_env_overrides(gpu_sm=gpu_sm, cuda_version=cuda_version, build_env=build_env, venv=venv)

    if not is_windows():
        return build_env, None

    native_env = dict(build_env)
    diagnostics: dict[str, object] = {"platform": "windows-native-cuda"}
    venv_bin_dir = venv_bin(venv, "python").parent
    native_env = prepend_directory_to_path(native_env, venv_bin_dir)

    toolkit_root = resolve_cuda_toolkit_root(cuda_version, env=native_env)
    diagnostics["cuda_toolkit_root_candidates"] = [str(path) for path in candidate_cuda_toolkit_roots(cuda_version, env=native_env)]
    if toolkit_root is None:
        raise RuntimeError(
            "Windows native CUDA builds require the CUDA Toolkit. Could not resolve CUDA_HOME/CUDA_PATH. "
            "Install the NVIDIA CUDA Toolkit matching your PyTorch CUDA wheel, or set MODLY_TRELLIS_TEXT_CUDA_TOOLKIT_ROOT."
        )

    native_env["CUDA_HOME"] = str(toolkit_root)
    native_env["CUDA_PATH"] = str(toolkit_root)
    native_env["CUDACXX"] = str(toolkit_root / "bin" / "nvcc.exe")
    prepend_env_path(native_env, "PATH", venv_bin_dir, toolkit_root / "bin")
    include_dir = toolkit_root / "include"
    if include_dir.exists():
        prepend_env_path(native_env, "INCLUDE", include_dir)
    library_dirs = cuda_toolkit_library_dirs(toolkit_root)
    if library_dirs:
        prepend_env_path(native_env, "LIB", *library_dirs)
        prepend_env_path(native_env, "LIBPATH", *library_dirs)

    native_env, msvc_diagnostics = resolve_windows_msvc_env(native_env)
    # vcvars64.bat rewrites PATH/INCLUDE/LIB. Re-prepend the selected extension
    # venv and CUDA toolkit paths afterwards so PyTorch CUDA extensions compile
    # against the same toolkit selected by setup.py.
    native_env["CUDA_HOME"] = str(toolkit_root)
    native_env["CUDA_PATH"] = str(toolkit_root)
    native_env["CUDACXX"] = str(toolkit_root / "bin" / "nvcc.exe")
    prepend_env_path(native_env, "PATH", venv_bin_dir, toolkit_root / "bin")
    if include_dir.exists():
        prepend_env_path(native_env, "INCLUDE", include_dir)
    if library_dirs:
        prepend_env_path(native_env, "LIB", *library_dirs)
        prepend_env_path(native_env, "LIBPATH", *library_dirs)
    diagnostics.update(
        {
            "cuda_toolkit_root": str(toolkit_root),
            "CUDA_HOME": native_env["CUDA_HOME"],
            "CUDA_PATH": native_env["CUDA_PATH"],
            "CUDACXX": native_env["CUDACXX"],
            "msvc": msvc_diagnostics,
        }
    )
    return native_env, diagnostics


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
    diagnostics["CUDA_PATH"] = source_env["CUDA_PATH"]
    diagnostics["CUDACXX"] = source_env["CUDACXX"]
    diagnostics["source_build_hotfixes"] = [
        "patch installed cumm/common.py on Linux ARM64 so CUDA include/lib discovery honors CUDA_HOME/CUDA_PATH before /usr/local/cuda"
    ]
    return source_env, diagnostics


def patch_installed_cumm_cuda_discovery(venv: Path) -> None:
    """Patch cumm's CUDA discovery to honor the selected CUDA toolkit root.

    cumm v0.7.11 can discover `/usr/local/cuda` even when setup.py selected a
    versioned toolkit such as `/usr/local/cuda-12.8`. On ARM64 that can mix a
    CUDA 12.8 nvcc with headers from a different toolkit and fail while building
    spconv with errors such as:

        macro "__cudaLaunch" requires 2 arguments, but only 1 given

    The full TRELLIS extension carries the same hotfix. The text-only extension
    needs it as well because it still depends on spconv.
    """

    cumm_common = Path(
        subprocess.check_output(
            [str(venv_bin(venv, "python")), "-c", "import cumm.common; print(cumm.common.__file__)"],
            text=True,
        ).strip()
    )
    original = cumm_common.read_text(encoding="utf-8")
    if CUMM_CUDA_DISCOVERY_PATCH_MARKER in original:
        print(f"[setup] cumm CUDA discovery patch already present at {cumm_common}")
        return

    old = """        else:\n            try:\n                nvcc_path = subprocess.check_output([\"which\", \"nvcc\"\n                                                    ]).decode(\"utf-8\").strip()\n                lib = Path(nvcc_path).parent.parent / \"lib\"\n                include = Path(nvcc_path).parent.parent / \"targets/x86_64-linux/include\"\n                if lib.exists() and include.exists():\n                    if (lib / \"libcudart.so\").exists() and (include / \"cuda.h\").exists():\n                        # should be nvidia conda package\n                        _CACHED_CUDA_INCLUDE_LIB = ([include], lib)\n                        return _CACHED_CUDA_INCLUDE_LIB\n            except:\n                pass \n\n            linux_cuda_root = Path(\"/usr/local/cuda\")\n            include = linux_cuda_root / f\"include\"\n            lib64 = linux_cuda_root / f\"lib64\"\n            assert linux_cuda_root.exists(), f\"can't find cuda in {linux_cuda_root} install via cuda installer or conda first.\"\n"""
    new = f"""        else:\n            # {CUMM_CUDA_DISCOVERY_PATCH_MARKER}\n            try:\n                nvcc_path = subprocess.check_output([\"which\", \"nvcc\"\n                                                    ]).decode(\"utf-8\").strip()\n                linux_cuda_root = Path(nvcc_path).parent.parent\n                include_candidates = [\n                    linux_cuda_root / \"targets/x86_64-linux/include\",\n                    linux_cuda_root / \"targets/aarch64-linux/include\",\n                    linux_cuda_root / \"targets/sbsa-linux/include\",\n                    linux_cuda_root / \"include\",\n                ]\n                lib_candidates = [\n                    linux_cuda_root / \"lib\",\n                    linux_cuda_root / \"lib64\",\n                    linux_cuda_root / \"targets/x86_64-linux/lib\",\n                    linux_cuda_root / \"targets/aarch64-linux/lib\",\n                    linux_cuda_root / \"targets/sbsa-linux/lib\",\n                ]\n                for include in include_candidates:\n                    for lib in lib_candidates:\n                        if (lib / \"libcudart.so\").exists() and (include / \"cuda.h\").exists():\n                            # should be nvidia conda package or an explicitly selected toolkit root\n                            _CACHED_CUDA_INCLUDE_LIB = ([include], lib)\n                            return _CACHED_CUDA_INCLUDE_LIB\n            except:\n                pass \n\n            linux_cuda_roots = []\n            for env_name in (\"CUDA_HOME\", \"CUDA_PATH\"):\n                env_value = os.getenv(env_name)\n                if env_value:\n                    linux_cuda_roots.append(Path(env_value))\n            linux_cuda_roots.append(Path(\"/usr/local/cuda\"))\n            for linux_cuda_root in linux_cuda_roots:\n                include_candidates = [\n                    linux_cuda_root / \"include\",\n                    linux_cuda_root / \"targets/x86_64-linux/include\",\n                    linux_cuda_root / \"targets/aarch64-linux/include\",\n                    linux_cuda_root / \"targets/sbsa-linux/include\",\n                ]\n                lib_candidates = [\n                    linux_cuda_root / \"lib64\",\n                    linux_cuda_root / \"lib\",\n                    linux_cuda_root / \"targets/x86_64-linux/lib\",\n                    linux_cuda_root / \"targets/aarch64-linux/lib\",\n                    linux_cuda_root / \"targets/sbsa-linux/lib\",\n                ]\n                for include in include_candidates:\n                    for lib64 in lib_candidates:\n                        if (lib64 / \"libcudart.so\").exists() and (include / \"cuda.h\").exists():\n                            _CACHED_CUDA_INCLUDE_LIB = ([include], lib64)\n                            return _CACHED_CUDA_INCLUDE_LIB\n            linux_cuda_root = Path(\"/usr/local/cuda\")\n            include = linux_cuda_root / f\"include\"\n            lib64 = linux_cuda_root / f\"lib64\"\n            assert linux_cuda_root.exists(), f\"can't find cuda in {{linux_cuda_root}} install via cuda installer or conda first.\"\n"""
    if old not in original:
        raise RuntimeError(f"Unable to patch cumm CUDA discovery at {cumm_common}; upstream layout changed.")
    cumm_common.write_text(original.replace(old, new, 1), encoding="utf-8")
    print(f"[setup] Patched cumm CUDA discovery at {cumm_common} to honor explicit CUDA toolkit roots on Linux ARM64.")


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print("[setup] $", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd else None)


def pip(venv: Path, *args: str, env: dict[str, str] | None = None) -> None:
    # Always invoke pip through the venv Python executable. On Windows, running
    # `venv\\Scripts\\pip.exe install --upgrade pip ...` can fail because pip is
    # trying to replace the wrapper currently executing. `python -m pip` is the
    # supported cross-platform form and also works on Linux.
    run([str(venv_bin(venv, "python")), "-m", "pip", *args], env=env)


def vendor_sources_ready(ext_dir: Path) -> bool:
    return all((ext_dir / relative_path).exists() for relative_path in VENDOR_REQUIRED_PATHS)


def ensure_vendor_sources(ext_dir: Path, venv: Path) -> None:
    if vendor_sources_ready(ext_dir):
        print("[setup] vendor/ already contains TRELLIS text runtime sources.")
        return

    build_vendor = ext_dir / "build_vendor.py"
    if not build_vendor.exists():
        raise RuntimeError(
            f"Missing {build_vendor}. Cannot populate vendor/ with official TRELLIS text runtime sources. "
            "Reinstall the extension from the GitHub repository."
        )

    print("[setup] Populating vendor/ with official TRELLIS text runtime sources ...")
    try:
        run([str(venv_bin(venv, "python")), str(build_vendor)], cwd=ext_dir)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to populate vendor/ with official TRELLIS text runtime sources. "
            "Check network access to GitHub/PyPI and rerun setup."
        ) from exc

    if not vendor_sources_ready(ext_dir):
        missing = [str(path) for path in VENDOR_REQUIRED_PATHS if not (ext_dir / path).exists()]
        raise RuntimeError("vendor/ was populated but required runtime sources are still missing: " + ", ".join(missing))


def pip_install(venv: Path, *packages: str, env: dict[str, str] | None = None, no_build_isolation: bool = False, no_deps: bool = False) -> None:
    cmd = ["install"]
    if no_build_isolation:
        cmd.append("--no-build-isolation")
    if no_deps:
        cmd.append("--no-deps")
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


def smoke_check_native_wheels(venv: Path, *, env: dict[str, str] | None = None) -> None:
    python(
        venv,
        "-c",
        "import torch; import nvdiffrast.torch; import diff_gaussian_rasterization; print('[setup] native wheel imports OK:', torch.__version__)",
        env=env,
    )


def smoke_check_spconv(venv: Path, *, env: dict[str, str] | None = None) -> None:
    # Import torch first so Windows registers PyTorch/CUDA DLL directories before
    # spconv loads its native extension modules. The runtime generator does the
    # same before importing native dependencies; setup's smoke test must match it
    # or it can reject a valid wheel with a DLL-load false negative.
    python(
        venv,
        "-c",
        "import warnings; warnings.filterwarnings('ignore', category=FutureWarning, module=r'spconv(\\.|$).*'); import torch; import spconv.pytorch as spconv; print('[setup] spconv import OK:', getattr(spconv, '__version__', 'unknown'))",
        env=env,
    )


def smoke_check_torch_stack(venv: Path, torch_packages: list[str], *, env: dict[str, str] | None = None) -> None:
    expected_torch = package_version(torch_packages, "torch")
    expected_torchvision = package_version(torch_packages, "torchvision")
    code = (
        "import torch, torchvision; "
        f"assert torch.__version__.split('+')[0] == '{expected_torch}', torch.__version__; "
        f"assert torchvision.__version__.split('+')[0] == '{expected_torchvision}', torchvision.__version__; "
        "print('[setup] torch stack OK:', torch.__version__, torchvision.__version__)"
    )
    python(venv, "-c", code, env=env)


def candidate_prebuilt_spconv_tags(cuda_tag: str) -> list[str]:
    if is_windows() and sys.version_info >= (3, 12):
        # PyPI publishes spconv-cu118 cp312-win_amd64 wheels, but not
        # spconv-cu120 cp312-win_amd64 wheels. Do not try tags that cannot
        # satisfy the current Windows embedded Python version.
        return ["cu118"]
    return [tag for tag in (cuda_tag, *KNOWN_PREBUILT_SPCONV_CUDA_TAGS) if tag in KNOWN_PREBUILT_SPCONV_CUDA_TAGS]


def install_prebuilt_spconv(venv: Path, cuda_tag: str) -> None:
    fallbacks = candidate_prebuilt_spconv_tags(cuda_tag)
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
    raise RuntimeError(
        "Failed to install a prebuilt spconv wheel. "
        f"Tried: {', '.join(tried)}. "
        "The upstream spconv project does not currently publish wheels for every CUDA tag/Python version."
    ) from last_error


def install_spconv_from_source(venv: Path, gpu_sm: int, cuda_version: int, build_env: dict[str, str]) -> None:
    source_env, diagnostics = source_build_env_overrides(gpu_sm=gpu_sm, cuda_version=cuda_version, build_env=build_env, venv=venv)
    print(f"[setup] Linux ARM64 source build for cumm/spconv: {json.dumps(diagnostics, indent=2)}")
    uninstall_packages(venv, "spconv", "cumm")
    install_packages_with_diagnostics(venv, "spconv-build-prereqs", "pccm/ccimport/pybind11/fire", "pccm>=0.4.16", "ccimport>=0.4.4", "pybind11>=2.6.0", "fire", env=source_env)
    with tempfile.TemporaryDirectory(prefix="trellis-text-spconv-") as tmp:
        tmpdir = Path(tmp)
        install_from_repo(venv, tmpdir, "cumm", CUMM_SOURCE_REPO, ref=CUMM_SOURCE_REF, env=source_env, no_deps=True)
        patch_installed_cumm_cuda_discovery(venv)
        install_from_repo(venv, tmpdir, "spconv", SPCONV_SOURCE_REPO, ref=SPCONV_SOURCE_REF, env=source_env, no_deps=True)
    smoke_check_spconv(venv, env=source_env)


def install_spconv(venv: Path, cuda_tag: str, gpu_sm: int, build_env: dict[str, str]) -> None:
    if is_linux_arm64():
        cuda_version = int(cuda_tag[2:]) if cuda_tag.startswith("cu") else 0
        install_spconv_from_source(venv, gpu_sm, cuda_version, build_env)
    else:
        install_prebuilt_spconv(venv, cuda_tag)


def install_attention_backend(venv: Path, plan: PlatformInstallPlan, torch_packages: list[str]) -> str:
    failures: list[str] = []
    for backend_name, requirement in resolve_attention_backends(plan, torch_packages):
        try:
            pip_install(
                venv,
                requirement,
                no_build_isolation=attention_backend_needs_no_build_isolation(backend_name, requirement),
                no_deps=backend_name == "xformers",
            )
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


def native_wheel_base_url() -> str:
    return os.environ.get(
        "MODLY_TRELLIS_TEXT_NATIVE_WHEEL_BASE_URL",
        f"https://github.com/{NATIVE_WHEEL_RELEASE_REPO}/releases/download/{NATIVE_WHEEL_RELEASE_TAG}",
    ).rstrip("/")


def native_wheel_urls(abi_tag: str) -> dict[str, str]:
    base_url = native_wheel_base_url()
    return {
        package_name: f"{base_url}/{metadata['filename'].format(abi=abi_tag)}"
        for package_name, metadata in NATIVE_WHEEL_FILENAMES.items()
    }


def try_install_prebuilt_native_wheels(venv: Path, torch_packages: list[str], cuda_tag: str) -> bool:
    if os.environ.get("MODLY_TRELLIS_TEXT_DISABLE_NATIVE_WHEELS") == "1":
        print("[setup] Prebuilt native wheels disabled by MODLY_TRELLIS_TEXT_DISABLE_NATIVE_WHEELS=1; falling back to source builds.")
        return False

    abi_tag = python_abi_tag()
    torch_version = package_version(torch_packages, "torch")
    torchvision_version = package_version(torch_packages, "torchvision")
    detected_platform = wheel_platform_tag()
    if not is_windows() or detected_platform != "win_amd64":
        print(f"[setup] No compatible native wheel strategy for platform={detected_platform}; source build fallback remains active.")
        return False
    if abi_tag is None:
        print(
            f"[setup] No compatible native wheels for Python ABI {sys.version_info.major}.{sys.version_info.minor}; "
            "supported ABIs are cp311/cp312. Falling back to source builds."
        )
        return False
    if (
        cuda_tag != NATIVE_WHEEL_SUPPORTED_CUDA_TAG
        or torch_version != NATIVE_WHEEL_SUPPORTED_TORCH
        or torchvision_version != NATIVE_WHEEL_SUPPORTED_TORCHVISION
    ):
        print(
            "[setup] No compatible prebuilt native wheels for "
            f"abi={abi_tag}, torch=={torch_version}, torchvision=={torchvision_version}, cuda_tag={cuda_tag}. "
            "Falling back to source builds that require CUDA Toolkit/MSVC on Windows."
        )
        return False

    urls = native_wheel_urls(abi_tag)
    print(f"[setup] Trying Windows native wheels from release tag {NATIVE_WHEEL_RELEASE_TAG}: {json.dumps(urls, indent=2)}")
    try:
        pip_install(venv, *urls.values(), no_deps=True)
        smoke_check_native_wheels(venv)
        print("[setup] Installed native TRELLIS wheels successfully; CUDA Toolkit/MSVC source build step is not required.")
        return True
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        print(
            "[setup] Prebuilt native wheel install failed; "
            "falling back to source builds that require CUDA Toolkit/MSVC on Windows. "
            f"Cause: {exc}"
        )
        uninstall_packages(venv, "nvdiffrast", "diff_gaussian_rasterization", "diff-gaussian-rasterization")
        return False


def describe_install_plan(gpu_sm: int, cuda_version: int) -> dict[str, object]:
    torch_pkgs, torch_index, cuda_tag = select_torch(gpu_sm, cuda_version)
    plan = plan_platform_install()
    description: dict[str, object] = {
        "platform": platform_label(),
        "plan": plan.name,
        "torch_packages": torch_pkgs,
        "torch_index": torch_index,
        "cuda_tag": cuda_tag,
        "spconv_strategy": "source-with-cumm-cuda-discovery-patch" if is_linux_arm64() else f"prebuilt-known-tags:{','.join(KNOWN_PREBUILT_SPCONV_CUDA_TAGS)}",
        "attention_backends": [f"{backend}:{requirement}" for backend, requirement in resolve_attention_backends(plan, torch_pkgs)],
        "native_from_git": {
            "nvdiffrast": f"{NVDIFFRAST_SOURCE_REPO}@{NVDIFFRAST_SOURCE_REF}",
            "diff_gaussian_rasterization": f"{MIP_SPLATTING_SOURCE_REPO}@{MIP_SPLATTING_SOURCE_REF}:{MIP_SPLATTING_DIFF_GAUSSIAN_SUBDIRECTORY}",
        },
        "native_wheels": {
            "enabled_by_default": True,
            "disable_env": "MODLY_TRELLIS_TEXT_DISABLE_NATIVE_WHEELS=1",
            "base_url_env": "MODLY_TRELLIS_TEXT_NATIVE_WHEEL_BASE_URL",
            "release_tag": NATIVE_WHEEL_RELEASE_TAG,
            "supported_platform": "win_amd64",
            "supported_abis": ["cp311", "cp312"],
            "supported_torch": NATIVE_WHEEL_SUPPORTED_TORCH,
            "supported_torchvision": NATIVE_WHEEL_SUPPORTED_TORCHVISION,
            "supported_cuda_tag": NATIVE_WHEEL_SUPPORTED_CUDA_TAG,
        },
        "excluded": ["TRELLIS.2 image/texturing", "o-voxel", "CuMesh", "DINOv3", "RMBG", "nvdiffrec"],
    }
    if is_linux_arm64():
        _, diagnostics = source_build_env_overrides(gpu_sm=gpu_sm, cuda_version=cuda_version)
        description["source_build_env"] = diagnostics
    elif is_windows():
        description["native_build_env"] = {
            "strategy": "windows-msvc-cuda-env",
            "requires": ["Visual Studio Build Tools 2022 with Desktop development with C++", "NVIDIA CUDA Toolkit"],
        }
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
    ensure_vendor_sources(ext_dir, venv)

    torch_pkgs, torch_index, cuda_tag = select_torch(gpu_sm, cuda_version)
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)
    smoke_check_torch_stack(venv, torch_pkgs)
    install_python_runtime_dependencies(venv)
    install_spconv(venv, cuda_tag, gpu_sm, build_env)
    chosen_attention_backend = install_attention_backend(venv, plan, torch_pkgs)
    smoke_check_torch_stack(venv, torch_pkgs)
    print(f"[setup] Selected sparse attention backend: {chosen_attention_backend}")

    installed_native_wheels = try_install_prebuilt_native_wheels(venv, torch_pkgs, cuda_tag)
    if installed_native_wheels:
        print("[setup] Native TRELLIS postprocessing dependencies satisfied by Windows wheels.")
        print("[setup] Done. Extension venv is ready at:", venv)
        print("[setup] First runtime load still requires Hugging Face access for microsoft/TRELLIS-text-xlarge and hidden CLIP assets.")
        return

    native_build_env, native_diagnostics = resolve_native_build_env(venv, gpu_sm=gpu_sm, cuda_version=cuda_version, build_env=build_env)
    if native_diagnostics and native_diagnostics.get("cuda_toolkit_root"):
        print(f"[setup] Steering native source builds to CUDA toolkit root: {native_diagnostics['cuda_toolkit_root']}")
    if native_diagnostics and native_diagnostics.get("msvc"):
        print(f"[setup] Windows MSVC native build env: {json.dumps(native_diagnostics['msvc'], indent=2)}")

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
