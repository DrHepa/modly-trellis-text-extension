"""
Text-only TRELLIS extension for Modly.

Runtime shape:
    prompt -> TrellisTextTo3DPipeline -> mesh + gaussian -> textured GLB

Only pure-Python upstream sources live under vendor/. Native CUDA packages such
as nvdiffrast and diff_gaussian_rasterization must resolve from the extension
venv prepared by setup.py, not from vendor/.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from services.generators.base import BaseGenerator, GenerationCancelled, smooth_progress


_EXTENSION_DIR = Path(__file__).parent
_VENDOR_DIR = _EXTENSION_DIR / "vendor"
_TEXT_PIPELINE_CONFIG_FILE = "pipeline.text-localized.json"
_TEXT_AUX_WEIGHTS_DIR = "localized-aux-weights"

_NATIVE_VENDOR_OVERLAPS = {
    "nvdiffrast",
    "diff_gaussian_rasterization",
}


TEXT_TO_MESH_PARAMS_SCHEMA = [
    {"id": "prompt", "label": "Prompt", "type": "text", "default": ""},
    {"id": "sparse_steps", "label": "Sparse Structure Steps", "type": "int", "default": 12, "min": 1, "max": 50},
    {"id": "slat_steps", "label": "Structured Latent Steps", "type": "int", "default": 12, "min": 1, "max": 50},
    {"id": "sparse_cfg", "label": "Sparse CFG", "type": "float", "default": 7.5, "min": 0.0, "max": 20.0},
    {"id": "slat_cfg", "label": "Structured Latent CFG", "type": "float", "default": 7.5, "min": 0.0, "max": 20.0},
    {
        "id": "texture_size",
        "label": "Texture Size",
        "type": "select",
        "options": [{"value": 1024, "label": "1024"}, {"value": 2048, "label": "2048"}, {"value": 4096, "label": "4096"}],
        "default": 1024,
    },
    {"id": "simplify", "label": "Simplify Ratio", "type": "float", "default": 0.95, "min": 0.0, "max": 1.0},
    {"id": "seed", "label": "Seed", "type": "int", "default": 42, "min": 0, "max": 2147483647},
]


def module_spec_origin(module_name: str) -> str | None:
    spec = importlib.util.find_spec(module_name)
    return getattr(spec, "origin", None) if spec is not None else None


def vendor_path_fragment() -> str:
    return str(_VENDOR_DIR)


def reject_native_vendor_overlaps() -> None:
    overlaps = sorted(name for name in _NATIVE_VENDOR_OVERLAPS if (_VENDOR_DIR / name).exists())
    if overlaps:
        raise RuntimeError(
            "[TrellisTextGenerator] Native packages must not be vendored: "
            + ", ".join(overlaps)
            + ". Remove them from vendor/ and reinstall through setup.py."
        )


class TrellisTextGenerator(BaseGenerator):
    MODEL_ID = "trellis-text"
    DISPLAY_NAME = "TRELLIS Text"
    VRAM_GB = 24

    def is_downloaded(self) -> bool:
        return (self.model_dir / "pipeline.json").exists()

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._auto_download()

        self._setup_env()
        self._setup_vendor()

        from trellis.pipelines import TrellisTextTo3DPipeline

        config_file = self._prepare_text_pipeline_config(self.model_dir).name
        print(f"[TrellisTextGenerator] Loading text pipeline from {self.model_dir}...")
        pipe = TrellisTextTo3DPipeline.from_pretrained(str(self.model_dir), config_file=config_file)
        pipe.cuda()
        self._model = pipe
        print("[TrellisTextGenerator] Loaded on CUDA.")

    @classmethod
    def params_schema(cls) -> list[dict[str, Any]]:
        return TEXT_TO_MESH_PARAMS_SCHEMA

    @classmethod
    def capability_params_schema(cls, node_id: str) -> list[dict[str, Any]]:
        if node_id != "text-to-mesh":
            raise RuntimeError(f"[TrellisTextGenerator] Unsupported node '{node_id}'. Only text-to-mesh is available.")
        return TEXT_TO_MESH_PARAMS_SCHEMA

    def generate(
        self,
        image_bytes: bytes | None = None,
        params: dict[str, Any] | None = None,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        del image_bytes
        return self._generate_text_to_mesh(params or {}, progress_cb, cancel_event)

    def _setup_env(self) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("SPARSE_CONV_BACKEND", "spconv")

        if importlib.util.find_spec("xformers") is not None:
            os.environ.setdefault("ATTN_BACKEND", "xformers")
            os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
        elif importlib.util.find_spec("flash_attn") is not None:
            os.environ.setdefault("ATTN_BACKEND", "flash_attn")
            os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")

    def _setup_vendor(self) -> None:
        if not _VENDOR_DIR.exists():
            raise RuntimeError(
                f"[TrellisTextGenerator] vendor/ directory not found at {_VENDOR_DIR}. "
                "Run 'python build_vendor.py' from this extension directory."
            )

        reject_native_vendor_overlaps()

        import torch  # noqa: F401  # registers CUDA/DLL paths before native imports on Windows

        self._require_runtime_dependency("spconv", "spconv")
        self._require_runtime_dependency("nvdiffrast", "nvdiffrast", allow_vendor=False)
        self._require_runtime_dependency("diff_gaussian_rasterization", "diff_gaussian_rasterization", allow_vendor=False)
        self._require_runtime_dependency("xatlas", "xatlas")
        self._require_runtime_dependency("pyvista", "pyvista")
        self._require_runtime_dependency("igraph", "igraph")
        self._require_runtime_dependency("pymeshfix", "pymeshfix")

        if importlib.util.find_spec("xformers") is None and importlib.util.find_spec("flash_attn") is None:
            raise RuntimeError("[TrellisTextGenerator] Missing attention backend. Install xformers or flash-attn via setup.py.")

        if vendor_path_fragment() not in sys.path:
            sys.path.append(vendor_path_fragment())

        for module_name in _NATIVE_VENDOR_OVERLAPS:
            origin = module_spec_origin(module_name)
            if origin and vendor_path_fragment() in origin:
                raise RuntimeError(
                    f"[TrellisTextGenerator] '{module_name}' resolved from vendor/ instead of the extension venv. "
                    "Remove native vendored files and reinstall with setup.py."
                )

        try:
            from trellis.pipelines import TrellisTextTo3DPipeline  # noqa: F401
            from trellis.utils import postprocessing_utils  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(f"[TrellisTextGenerator] vendor/ is incomplete: {exc}") from exc

    def _require_runtime_dependency(self, module_name: str, package_name: str, *, allow_vendor: bool = True) -> None:
        origin = module_spec_origin(module_name)
        if origin is None:
            raise RuntimeError(
                f"[TrellisTextGenerator] Missing runtime dependency '{module_name}'. "
                f"Re-run setup.py so '{package_name}' is installed in the extension venv."
            )
        if not allow_vendor and vendor_path_fragment() in origin:
            raise RuntimeError(
                f"[TrellisTextGenerator] Runtime dependency '{module_name}' resolved from vendor/. "
                f"It must be installed as '{package_name}' in the extension venv."
            )

    def _prepare_text_pipeline_config(self, model_dir: Path) -> Path:
        source_config = model_dir / "pipeline.json"
        if not source_config.exists():
            raise RuntimeError(f"[TrellisTextGenerator] Text pipeline config is missing at {source_config}.")

        config = json.loads(source_config.read_text(encoding="utf-8"))
        args = config.get("args")
        if not isinstance(args, dict):
            raise RuntimeError("[TrellisTextGenerator] Text pipeline config is missing an 'args' object.")

        model_refs = args.get("models")
        if not isinstance(model_refs, dict):
            raise RuntimeError("[TrellisTextGenerator] Text pipeline config is missing an 'args.models' object.")

        localized_model_refs = dict(model_refs)
        localized_any = False
        for model_name, model_ref in model_refs.items():
            if not isinstance(model_ref, str):
                continue
            localized_ref = self._localize_auxiliary_model_ref(model_dir, model_ref)
            if localized_ref != model_ref:
                localized_model_refs[model_name] = localized_ref
                localized_any = True

        if localized_any:
            args["models"] = localized_model_refs

        localized_config = model_dir / _TEXT_PIPELINE_CONFIG_FILE
        payload = json.dumps(config, indent=4, ensure_ascii=False) + "\n"
        if not localized_config.exists() or localized_config.read_text(encoding="utf-8") != payload:
            localized_config.write_text(payload, encoding="utf-8")
        return localized_config

    def _localize_auxiliary_model_ref(self, owner_dir: Path, model_ref: str) -> str:
        if "/" not in model_ref:
            return model_ref

        ref_parts = model_ref.split("/")
        if len(ref_parts) < 3:
            return model_ref

        repo_id = "/".join(ref_parts[:2])
        relative_model_path = "/".join(ref_parts[2:])
        if not relative_model_path:
            return model_ref

        from huggingface_hub import hf_hub_download

        local_base = owner_dir / _TEXT_AUX_WEIGHTS_DIR / ref_parts[0] / ref_parts[1] / relative_model_path
        local_base.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".json", ".safetensors"):
            localized_file = local_base.with_suffix(suffix)
            if localized_file.exists():
                continue
            downloaded_file = Path(hf_hub_download(repo_id, f"{relative_model_path}{suffix}"))
            shutil.copy2(downloaded_file, localized_file)

        return local_base.relative_to(owner_dir).as_posix()

    def _normalize_prompt(self, params: dict[str, Any]) -> str:
        for key in ("prompt", "text", "input_text"):
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.strip().split())
        raise RuntimeError("[TrellisTextGenerator] text-to-mesh requires a non-empty params.prompt value.")

    def _generate_text_to_mesh(
        self,
        params: dict[str, Any],
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        from trellis.utils import postprocessing_utils

        prompt = self._normalize_prompt(params)
        sparse_steps = int(params.get("sparse_steps", 12))
        slat_steps = int(params.get("slat_steps", 12))
        sparse_cfg = float(params.get("sparse_cfg", 7.5))
        slat_cfg = float(params.get("slat_cfg", 7.5))
        texture_size = int(params.get("texture_size", 1024))
        simplify = float(params.get("simplify", 0.95))
        seed = int(params.get("seed", 42))

        self._report(progress_cb, 5, "Validating prompt...")
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 10, "Generating native TRELLIS text mesh...")
        outputs = self._run_with_smoothed_progress(
            progress_cb,
            start=10,
            end=88,
            label="Generating native TRELLIS text mesh...",
            run=lambda: self._model.run(
                prompt,
                seed=seed,
                sparse_structure_sampler_params={"steps": sparse_steps, "cfg_strength": sparse_cfg},
                slat_sampler_params={"steps": slat_steps, "cfg_strength": slat_cfg},
                formats=["mesh", "gaussian"],
            ),
        )
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 92, "Baking textures & exporting GLB...")
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=simplify,
            texture_size=texture_size,
            verbose=False,
        )
        self._check_cancelled(cancel_event)

        output_path = self._next_output_path()
        glb.export(str(output_path))
        self._report(progress_cb, 100, "Done")
        return output_path

    def _report(self, progress_cb: Optional[Callable[[int, str], None]], value: int, label: str) -> None:
        if progress_cb:
            progress_cb(value, label)

    def _check_cancelled(self, cancel_event: Optional[threading.Event]) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled()

    def _run_with_smoothed_progress(
        self,
        progress_cb: Optional[Callable[[int, str], None]],
        *,
        start: int,
        end: int,
        label: str,
        run: Callable[[], object],
    ) -> object:
        stop_evt = threading.Event()
        if progress_cb:
            thread = threading.Thread(target=smooth_progress, args=(progress_cb, start, end, label, stop_evt, 5.0), daemon=True)
            thread.start()
        try:
            return run()
        finally:
            stop_evt.set()

    def _next_output_path(self) -> Path:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        return self.outputs_dir / f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
