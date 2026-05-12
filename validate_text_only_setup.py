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


def validate_vendor_placeholder() -> None:
    require((ROOT / "vendor" / ".gitkeep").exists(), "vendor/.gitkeep placeholder is required")


def main() -> None:
    validate_manifest()
    validate_no_removed_runtime_code()
    validate_setup_exclusions()
    validate_vendor_placeholder()
    print("Static text-only validation passed.")


if __name__ == "__main__":
    main()
