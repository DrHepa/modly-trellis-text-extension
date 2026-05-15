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
    require(isinstance(nodes, list) and len(nodes) == 3, "manifest must expose Base, Large, and XL text nodes")

    expected_nodes = {
        "text-to-mesh-base": ("microsoft/TRELLIS-text-base", "text-base"),
        "text-to-mesh-large": ("microsoft/TRELLIS-text-large", "text-large"),
        "text-to-mesh": ("microsoft/TRELLIS-text-xlarge", "text-xlarge"),
    }
    actual_ids = {node.get("id") for node in nodes}
    require(actual_ids == set(expected_nodes), f"manifest nodes mismatch: {sorted(actual_ids)}")

    reference_param_ids: list[str] | None = None
    for node in nodes:
        node_id = node["id"]
        expected_repo, expected_owner = expected_nodes[node_id]
        require(node.get("capability_id") == node_id, f"{node_id} capability_id must match node id")
        require(node["input"] == "image", f"{node_id} input must remain image for upstream Modly model-node compatibility")
        require(node["output"] == "mesh", f"{node_id} output must remain mesh")
        require(node["hf_repo"] == expected_repo, f"{node_id} hf_repo must be {expected_repo}")
        require(node["weight_owner_id"] == expected_owner, f"{node_id} weight_owner_id must be {expected_owner}")
        require(node["download_check"] == "pipeline.json", f"{node_id} download_check must be pipeline.json")
        params = {param["id"]: param for param in node.get("params_schema", [])}
        require(params.get("prompt", {}).get("type") == "string", f"{node_id} prompt parameter must use upstream-supported string type")
        param_ids = [param["id"] for param in node.get("params_schema", [])]
        if reference_param_ids is None:
            reference_param_ids = param_ids
        require(param_ids == reference_param_ids, f"{node_id} must share the same params_schema order as the other text nodes")


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
    require("XFORMERS_BY_TORCH_VERSION" in setup and "xformers==0.0.30" in setup, "setup.py must pin xformers to selected torch versions")
    require("resolve_attention_backends" in setup, "setup.py must resolve attention backend pins from selected torch")
    require("no_deps=backend_name == \"xformers\"" in setup, "setup.py must install xformers without dependencies so torch is not replaced")
    require("smoke_check_torch_stack" in setup, "setup.py must verify torch/torchvision versions after dependency installs")
    require("try_install_prebuilt_native_wheels" in setup, "setup.py must attempt prebuilt Windows native wheels before source builds")
    require("--force-reinstall" in setup, "setup.py must force reinstall native wheels so repaired installs replace broken release wheels")
    require("MODLY_TRELLIS_TEXT_DISABLE_NATIVE_WHEELS" in setup, "setup.py must support disabling prebuilt native wheels")
    require("MODLY_TRELLIS_TEXT_NATIVE_WHEEL_BASE_URL" in setup, "setup.py must support overriding the native wheel release base URL")
    require("smoke_check_native_wheels" in setup, "setup.py must smoke-check imported native wheels")
    require("native-wheels-torch270-cu128-v2" in setup, "setup.py must target the widened-architecture native wheel release tag")
    require("candidate_prebuilt_spconv_tags" in setup and "return [\"cu118\"]" in setup, "setup.py must avoid unavailable spconv-cu120 cp312 Windows wheels")
    require("warnings.filterwarnings('ignore', category=FutureWarning" in setup, "spconv smoke check must suppress upstream FutureWarning noise")
    require("import torch; import spconv.pytorch as spconv" in setup, "spconv smoke check must import torch before spconv for Windows DLL paths")

    generator = (ROOT / "generator.py").read_text(encoding="utf-8")
    require("warnings.filterwarnings" in generator and "spconv" in generator, "generator must suppress spconv FutureWarnings at runtime")
    require("SPCONV_ALGO" in generator and "native" in generator, "generator must default spconv to native algo for one-off Modly runs")
    require(
        setup.index("install_spconv(venv") < setup.index("native_build_env, native_diagnostics = resolve_native_build_env"),
        "setup.py must install wheel-first dependencies before Windows native compiler preflight",
    )
    require(
        setup.index("try_install_prebuilt_native_wheels(venv, torch_pkgs, cuda_tag)") < setup.index("native_build_env, native_diagnostics = resolve_native_build_env"),
        "setup.py must try native wheels before resolving source-build CUDA/MSVC env",
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
    require("text-to-mesh-base" in readme and "text-to-mesh-large" in readme and "microsoft/TRELLIS-text-xlarge" in readme, "README must document all TRELLIS text model nodes")
    require("__cudaLaunch" in readme, "README must document the ARM64 CUDA toolkit mismatch failure mode")
    require("python -m pip" in readme, "README must document why setup uses python -m pip")
    require("Visual Studio Build Tools 2022" in readme, "README must document Windows native build prerequisites")
    require("vcvars64.bat" in readme, "README must mention automatic MSVC environment loading")
    require("build_vendor.py" in readme and "trellis.pipelines.TrellisTextTo3DPipeline" in readme, "README must document automatic vendor population")
    require("CUDA Toolkit 12.8" in readme and "spconv-cu128" in readme, "README must document Windows CUDA Toolkit and spconv-cu128 caveat")
    require("xformers" in readme and "cannot silently replace" in readme, "README must document pinned xformers/torch behavior")
    require("spconv-cu118" in readme and "FutureWarning" in readme, "README must document Windows spconv cp312/FutureWarning behavior")
    require("rembg" in readme and "TrellisTextTo3DPipeline" in readme, "README must document text-only TRELLIS pipeline export patching")
    require("open3d" in readme and "prompt-to-mesh" in readme, "README must document optional open3d vendoring patch")
    require(".trellis-text-only-v4" in readme, "README must document versioned vendor marker")
    require("pipeline.text-localized.json" in readme and "config_file" in readme, "README must document localized pipeline config support")
    require("kaolin" in readme, "README must document kaolin removal from vendored FlexiCubes")
    require("MODLY_TRELLIS_TEXT_DISABLE_NATIVE_WHEELS" in readme, "README must document native wheel opt-out")
    require("native-wheels/" in readme, "README must mention native wheel tooling docs")


def validate_native_wheels_tooling() -> None:
    required_files = [
        ROOT / "native-wheels" / "README.md",
        ROOT / "native-wheels" / "scripts" / "build-nvdiffrast.ps1",
        ROOT / "native-wheels" / "scripts" / "build-diff-gaussian.ps1",
        ROOT / "native-wheels" / "scripts" / "smoke-test.ps1",
        ROOT / "native-wheels" / "licenses" / "nvdiffrast-LICENSE.txt",
        ROOT / "native-wheels" / "licenses" / "diff-gaussian-rasterization-LICENSE.md",
        ROOT / ".github" / "workflows" / "build-native-windows-wheels.yml",
    ]
    for path in required_files:
        require(path.exists(), f"Missing required native wheel tooling file: {path.relative_to(ROOT)}")

    native_readme = (ROOT / "native-wheels" / "README.md").read_text(encoding="utf-8")
    require("CUDA Toolkit 12.8" in native_readme, "native-wheels README must require CUDA Toolkit 12.8")
    require("Visual Studio Build Tools 2022" in native_readme, "native-wheels README must document VS Build Tools 2022")
    require("native-wheels-torch270-cu128-v2" in native_readme, "native-wheels README must document the release tag")
    require("sm_75" in native_readme and "cudaErrorNoKernelImageForDevice" in native_readme, "native-wheels README must document widened CUDA architecture coverage")
    require("non-commercial" in native_readme and "research" in native_readme, "native-wheels README must document licensing limits")
    require("smoke-test.ps1" in native_readme, "native-wheels README must document smoke-test.ps1")
    require("build-native-windows-wheels.yml" in native_readme, "native-wheels README must document the GitHub Actions workflow")

    build_nvdiffrast = (ROOT / "native-wheels" / "scripts" / "build-nvdiffrast.ps1").read_text(encoding="utf-8")
    require("Set-StrictMode -Version Latest" in build_nvdiffrast, "build-nvdiffrast.ps1 must enable strict mode")
    require("$ErrorActionPreference = 'Stop'" in build_nvdiffrast, "build-nvdiffrast.ps1 must stop on errors")
    require("Invoke-Expression" not in build_nvdiffrast, "build-nvdiffrast.ps1 must not construct commands via Invoke-Expression")
    require("include\\cccl" in build_nvdiffrast and "$env:INCLUDE" in build_nvdiffrast, "build-nvdiffrast.ps1 must add Conda CUDA CCCL headers to INCLUDE")
    require("Ensure-CudaCcclHeaders" in build_nvdiffrast and "cuda-cccl_win-64" in build_nvdiffrast, "build-nvdiffrast.ps1 must normalize Conda CCCL nv headers into CUDA include")
    require("Ensure-CudaWindowsLibLayout" in build_nvdiffrast and "cudart.lib" in build_nvdiffrast, "build-nvdiffrast.ps1 must normalize Conda CUDA Windows library layout")
    require("pip wheel" in build_nvdiffrast and "--no-build-isolation" in build_nvdiffrast, "build-nvdiffrast.ps1 must build wheels via pip wheel --no-build-isolation")
    require("https://github.com/NVlabs/nvdiffrast.git" in build_nvdiffrast and "v0.4.0" in build_nvdiffrast, "build-nvdiffrast.ps1 must pin nvdiffrast source")
    require("6.1;7.5;8.0;8.6;8.9;9.0+PTX" in build_nvdiffrast, "build-nvdiffrast.ps1 must build portable Windows GPU architectures")

    build_diff = (ROOT / "native-wheels" / "scripts" / "build-diff-gaussian.ps1").read_text(encoding="utf-8")
    require("Set-StrictMode -Version Latest" in build_diff, "build-diff-gaussian.ps1 must enable strict mode")
    require("$ErrorActionPreference = 'Stop'" in build_diff, "build-diff-gaussian.ps1 must stop on errors")
    require("Invoke-Expression" not in build_diff, "build-diff-gaussian.ps1 must not construct commands via Invoke-Expression")
    require("include\\cccl" in build_diff and "$env:INCLUDE" in build_diff, "build-diff-gaussian.ps1 must add Conda CUDA CCCL headers to INCLUDE")
    require("Ensure-CudaCcclHeaders" in build_diff and "cuda-cccl_win-64" in build_diff, "build-diff-gaussian.ps1 must normalize Conda CCCL nv headers into CUDA include")
    require("Ensure-CudaWindowsLibLayout" in build_diff and "cudart.lib" in build_diff, "build-diff-gaussian.ps1 must normalize Conda CUDA Windows library layout")
    require("pip wheel" in build_diff and "--no-build-isolation" in build_diff, "build-diff-gaussian.ps1 must build wheels via pip wheel --no-build-isolation")
    require("https://github.com/autonomousvision/mip-splatting.git" in build_diff and "dda02ab5ecf45d6edb8c540d9bb65c7e451345a9" in build_diff, "build-diff-gaussian.ps1 must pin mip-splatting source")
    require("submodules/diff-gaussian-rasterization" in build_diff, "build-diff-gaussian.ps1 must build the diff-gaussian subdirectory")
    require("submodule update --init --recursive" in build_diff, "build-diff-gaussian.ps1 must initialize recursive submodules")
    require("6.1;7.5;8.0;8.6;8.9;9.0+PTX" in build_diff, "build-diff-gaussian.ps1 must build portable Windows GPU architectures")

    smoke_test = (ROOT / "native-wheels" / "scripts" / "smoke-test.ps1").read_text(encoding="utf-8")
    require("Invoke-Expression" not in smoke_test, "smoke-test.ps1 must not construct commands via Invoke-Expression")
    require("import torch; import nvdiffrast.torch; import diff_gaussian_rasterization" in smoke_test, "smoke-test.ps1 must validate torch and native imports")

    nvdiffrast_license = (ROOT / "native-wheels" / "licenses" / "nvdiffrast-LICENSE.txt").read_text(encoding="utf-8")
    require("Nvidia Source Code License" in nvdiffrast_license, "nvdiffrast license text must be complete")
    diff_license = (ROOT / "native-wheels" / "licenses" / "diff-gaussian-rasterization-LICENSE.md").read_text(encoding="utf-8")
    require("Gaussian-Splatting License" in diff_license, "diff-gaussian license text must be complete")

    workflow = (ROOT / ".github" / "workflows" / "build-native-windows-wheels.yml").read_text(encoding="utf-8")
    require("windows-2022" in workflow, "native wheel workflow must build on Windows")
    require("conda-incubator/setup-miniconda@v4" in workflow and "cuda-12.8.1" in workflow, "native wheel workflow must install CUDA Toolkit 12.8 packages from NVIDIA Conda")
    require("conda create -y -p $cudaEnv" in workflow, "native wheel workflow must install CUDA packages into an explicit prefix")
    require("cuda-cccl_win-64" in workflow, "native wheel workflow must install the real Windows CCCL header package")
    require("libcusparse-dev" in workflow and "libcublas-dev" in workflow, "native wheel workflow must install CUDA dev headers required by PyTorch")
    require("CUDA_HOME=$cudaRoot" in workflow and "CUDA_PATH=$cudaRoot" in workflow, "native wheel workflow must export CUDA_HOME/CUDA_PATH")
    require("ilammy/msvc-dev-cmd@v1" in workflow, "native wheel workflow must prepare MSVC developer shell")
    require("build-nvdiffrast.ps1" in workflow and "build-diff-gaussian.ps1" in workflow, "native wheel workflow must run both build scripts")
    require("smoke-test.ps1" in workflow, "native wheel workflow must smoke-test built wheels")
    require("gh release upload" in workflow, "native wheel workflow must support release upload")


def main() -> None:
    validate_manifest()
    validate_no_removed_runtime_code()
    validate_setup_exclusions()
    validate_build_vendor_text_only_patch()
    validate_vendor_placeholder()
    validate_readme()
    validate_native_wheels_tooling()
    print("Static text-only validation passed.")


if __name__ == "__main__":
    main()
