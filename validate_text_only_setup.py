"""Static validation for the text-only TRELLIS extension skeleton."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_manifest() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["author"] == "DrHepa", "manifest author must be DrHepa")
    require(manifest["generator_class"] == "TrellisTextGenerator", "manifest generator_class must be TrellisTextGenerator")
    require(manifest["source"] == "https://github.com/DrHepa/modly-trellis-text-extension", "manifest source mismatch")
    nodes = manifest.get("nodes")
    require(isinstance(nodes, list) and len(nodes) == 1, "manifest must expose exactly one node")
    node = nodes[0]
    require(node["id"] == "text-to-mesh", "only text-to-mesh node is allowed")
    require(node["capability_id"] == "text-to-mesh", "capability_id must be text-to-mesh")
    require(node["hf_repo"] == "microsoft/TRELLIS-text-xlarge", "hf_repo must be microsoft/TRELLIS-text-xlarge")
    require(node["download_check"] == "pipeline.json", "download_check must be pipeline.json")


def validate_no_removed_runtime_code() -> None:
    generator = (ROOT / "generator.py").read_text(encoding="utf-8")
    forbidden_generator_terms = [
        "Trellis2ImageTo3DPipeline",
        "Trellis2TexturingPipeline",
        "o_voxel",
        "cumesh",
        "DINOv3",
        "RMBG",
    ]
    for term in forbidden_generator_terms:
        require(term not in generator, f"generator.py must not contain removed runtime term: {term}")
    require("TrellisTextTo3DPipeline.from_pretrained" in generator, "generator must load the native text pipeline")
    require("postprocessing_utils.to_glb" in generator, "generator must export through official TRELLIS to_glb")


def validate_setup_exclusions() -> None:
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    require("CUMESH_SOURCE_REPO" not in setup, "setup.py must not install CuMesh")
    require("NVDIFFREC_SOURCE_REPO" not in setup, "setup.py must not install nvdiffrec")
    require("TRELLIS2_SOURCE_REPO" not in setup, "setup.py must not install TRELLIS.2/o-voxel")
    require("MIP_SPLATTING_DIFF_GAUSSIAN_SUBDIRECTORY" in setup, "setup.py must install diff_gaussian_rasterization")
    require("NVDIFFRAST_SOURCE_REPO" in setup, "setup.py must install nvdiffrast")
    require("patch_installed_cumm_cuda_discovery" in setup, "setup.py must patch cumm CUDA discovery before ARM64 spconv source builds")
    require("CUDA_HOME" in setup and "CUDA_PATH" in setup, "cumm patch must honor CUDA_HOME/CUDA_PATH")
    require("targets/aarch64-linux" in setup, "cumm patch must know ARM64 CUDA target include/lib paths")
    require('venv_bin(venv, "python")), "-m", "pip"' in setup, "setup.py must invoke pip via venv python -m pip")
    require('venv_bin(venv, "pip")' not in setup, "setup.py must not invoke pip.exe directly")
    require("resolve_windows_msvc_env" in setup, "setup.py must prepare MSVC env for Windows native CUDA builds")
    require("vswhere.exe" in setup and "vcvars64.bat" in setup, "setup.py must locate and load Visual Studio Build Tools on Windows")
    require("DISTUTILS_USE_SDK" in setup and "MSSdk" in setup, "setup.py must set Windows native build SDK flags")
    require("MODLY_TRELLIS_TEXT_CUDA_TOOLKIT_ROOT" in setup, "setup.py must allow explicit CUDA Toolkit override")
    require("ensure_vendor_sources(ext_dir, venv)" in setup, "setup.py must populate vendor/ during extension setup")
    require("build_vendor.py" in setup and "VENDOR_REQUIRED_PATHS" in setup, "setup.py must validate required TRELLIS vendor sources")
    require(".trellis-text-only-v4" in setup, "setup.py must require the versioned text-only vendor marker so stale vendor trees are rebuilt")
    require('"plyfile"' in setup, "setup.py must install plyfile for TRELLIS Gaussian PLY helpers")
    require("KNOWN_PREBUILT_SPCONV_CUDA_TAGS" in setup, "setup.py must restrict spconv fallback tags to known published wheels")
    require("import torch; import spconv.pytorch as spconv" in setup, "spconv smoke check must import torch before spconv for Windows DLL paths")
    require(
        setup.index("install_spconv(venv") < setup.index("native_build_env, native_diagnostics = resolve_native_build_env"),
        "setup.py must install wheel-first dependencies before Windows native compiler preflight",
    )


def validate_build_vendor_text_only_patch() -> None:
    build_vendor = (ROOT / "build_vendor.py").read_text(encoding="utf-8")
    require("patch_trellis_text_only_exports" in build_vendor, "build_vendor.py must patch TRELLIS pipeline exports for text-only runtime")
    require("from .trellis_text_to_3d import TrellisTextTo3DPipeline" in build_vendor, "build_vendor.py must export only TrellisTextTo3DPipeline")
    require("rembg" in build_vendor, "build_vendor.py must document why image pipeline imports are excluded")
    require("TEXT_ONLY_VENDOR_MARKER" in build_vendor, "build_vendor.py must stamp text-only vendor trees")
    require("patch_trellis_text_pipeline_optional_open3d" in build_vendor, "build_vendor.py must make open3d optional for prompt-to-mesh imports")
    require(".trellis-text-only-v4" in build_vendor, "build_vendor.py must stamp versioned text-only vendor trees")
    require("patch_trellis_config_file_support" in build_vendor, "build_vendor.py must patch TRELLIS from_pretrained config_file support")
    require("config_file: str = \"pipeline.json\"" in build_vendor, "build_vendor.py must add config_file defaults to TRELLIS pipeline loaders")
    require("patch_flexicubes_local_check_tensor" in build_vendor, "build_vendor.py must remove kaolin from vendored FlexiCubes")
    require("from kaolin.utils.testing import check_tensor" in build_vendor and "local check_tensor" in build_vendor, "build_vendor.py must patch kaolin check_tensor import")
    require("source.replace(\"import open3d as o3d" in build_vendor, "build_vendor.py must remove eager open3d imports from the text pipeline")
    require("open3d is required only for TRELLIS text run_variant" in build_vendor, "build_vendor.py must preserve a clear run_variant open3d error")


def validate_vendor_placeholder() -> None:
    require((ROOT / "vendor" / ".gitkeep").exists(), "vendor/.gitkeep placeholder is required")


def validate_readme() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    require("Modly TRELLIS Text Extension" in readme, "README must have a professional project title")
    require("__cudaLaunch" in readme, "README must document the ARM64 CUDA toolkit mismatch failure mode")
    require("python -m pip" in readme, "README must document why setup uses python -m pip")
    require("Visual Studio Build Tools 2022" in readme, "README must document Windows native build prerequisites")
    require("vcvars64.bat" in readme, "README must mention automatic MSVC environment loading")
    require("build_vendor.py" in readme and "trellis.pipelines.TrellisTextTo3DPipeline" in readme, "README must document automatic vendor population")
    require("CUDA Toolkit 12.8" in readme and "spconv-cu128" in readme, "README must document Windows CUDA Toolkit and spconv-cu128 caveat")
    require("rembg" in readme and "TrellisTextTo3DPipeline" in readme, "README must document text-only TRELLIS pipeline export patching")
    require("open3d" in readme and "prompt-to-mesh" in readme, "README must document optional open3d vendoring patch")
    require(".trellis-text-only-v4" in readme, "README must document versioned vendor marker")
    require("pipeline.text-localized.json" in readme and "config_file" in readme, "README must document localized pipeline config support")
    require("kaolin" in readme, "README must document kaolin removal from vendored FlexiCubes")


def main() -> None:
    validate_manifest()
    validate_no_removed_runtime_code()
    validate_setup_exclusions()
    validate_build_vendor_text_only_patch()
    validate_vendor_placeholder()
    validate_readme()
    print("Static text-only validation passed.")


if __name__ == "__main__":
    main()
