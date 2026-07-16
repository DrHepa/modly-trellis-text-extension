from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("trellis_text_setup_under_test", ROOT / "setup.py")
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery invariant
    raise RuntimeError("Unable to load setup.py for tests")
setup_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = setup_module
SPEC.loader.exec_module(setup_module)


REDIRECTORS = {
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PIP_USER",
    "PIP_TARGET",
    "PIP_PREFIX",
    "PIP_REQUIRE_VIRTUALENV",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
}


def contaminated_env() -> dict[str, str]:
    env = {name: f"poison-{name.lower()}" for name in REDIRECTORS}
    env.update(
        {
            "PATH": "toolchain-path",
            "CUDA_HOME": "cuda-home",
            "CUDA_PATH": "cuda-path",
            "HTTP_PROXY": "http://proxy.invalid",
            "SSL_CERT_FILE": "cert-bundle.pem",
        }
    )
    return env


def venv_python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def seed_probe_payload(
    executable: Path,
    prefix: Path,
    *,
    version: tuple[int, int] | None = None,
    base_prefix: Path | None = None,
) -> str:
    major_minor = version or (sys.version_info.major, sys.version_info.minor)
    return json.dumps(
        {
            "version": list(major_minor),
            "executable": str(executable),
            "prefix": str(prefix),
            "base_prefix": str(base_prefix or prefix),
        }
    )


class PythonSubprocessEnvironmentTests(unittest.TestCase):
    def test_sanitizer_removes_redirectors_and_preserves_runtime_environment(self) -> None:
        source = contaminated_env()

        cleaned = setup_module.clean_python_subprocess_env(source)

        self.assertTrue(REDIRECTORS.isdisjoint(cleaned))
        for name in ("PATH", "CUDA_HOME", "CUDA_PATH", "HTTP_PROXY", "SSL_CERT_FILE"):
            self.assertEqual(cleaned[name], source[name])
        self.assertTrue(REDIRECTORS.issubset(source), "the sanitizer must not mutate its input")

    def test_pip_and_python_helpers_sanitize_explicit_build_env(self) -> None:
        calls: list[tuple[list[str], dict[str, object]]] = []

        def capture_run(cmd: list[str], **kwargs: object) -> None:
            calls.append((list(cmd), dict(kwargs)))

        source = contaminated_env()
        venv = Path("extension-venv")
        with mock.patch.object(setup_module, "run", side_effect=capture_run):
            setup_module.pip(venv, "install", "example", env=source)
            setup_module.python(venv, "-c", "print('ok')", env=source)

        self.assertEqual(len(calls), 2)
        venv_python = str(setup_module.venv_bin(venv, "python"))
        self.assertEqual(calls[0][0], [venv_python, "-m", "pip", "install", "example"])
        self.assertEqual(calls[1][0], [venv_python, "-c", "print('ok')"])
        for _command, kwargs in calls:
            child_env = kwargs["env"]
            self.assertIsInstance(child_env, dict)
            self.assertTrue(REDIRECTORS.isdisjoint(child_env))
            self.assertEqual(child_env["CUDA_HOME"], source["CUDA_HOME"])
            self.assertEqual(child_env["HTTP_PROXY"], source["HTTP_PROXY"])


class VenvSeedSelectionTests(unittest.TestCase):
    def test_uses_existing_base_interpreter_for_modly_managed_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            managed = root / "dependencies" / "venv" / "Scripts" / "python.exe"
            base = root / "python-base" / "python.exe"
            managed.parent.mkdir(parents=True)
            base.parent.mkdir(parents=True)
            managed.touch()
            base.touch()

            with (
                mock.patch.object(setup_module.sys, "executable", str(managed)),
                mock.patch.object(setup_module.sys, "_base_executable", str(base), create=True),
            ):
                selected = setup_module.resolve_venv_seed_python(str(managed))

            self.assertEqual(selected, str(base))

    def test_falls_back_to_supplied_interpreter_when_base_is_not_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            managed = root / "dependencies" / "venv" / "Scripts" / "python.exe"
            other = root / "other" / "python.exe"
            missing_base = root / "missing-base" / "python.exe"
            managed.parent.mkdir(parents=True)
            other.parent.mkdir(parents=True)
            managed.touch()
            other.touch()

            cases = (
                (str(managed), str(missing_base)),
                (str(other), str(missing_base)),
                (str(other), str(managed)),
                (str(managed), str(managed)),
            )
            for supplied, base in cases:
                with self.subTest(supplied=supplied, base=base):
                    with (
                        mock.patch.object(setup_module.sys, "executable", str(managed)),
                        mock.patch.object(setup_module.sys, "_base_executable", base, create=True),
                    ):
                        selected = setup_module.resolve_venv_seed_python(supplied)
                    self.assertEqual(selected, supplied)


class ModlyLauncherCompatibilityTests(unittest.TestCase):
    def test_create_venv_uses_base_python_clean_env_and_clears_partial_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            managed = root / "dependencies" / "venv" / "Scripts" / "python.exe"
            base = root / "python-base" / "python.exe"
            target = root / "extensions" / "trellis-text" / "venv"
            managed.parent.mkdir(parents=True)
            base.parent.mkdir(parents=True)
            target.mkdir(parents=True)
            managed.touch()
            base.touch()
            stale_marker = target / "partial-ensurepip-install.txt"
            stale_marker.write_text("stale", encoding="utf-8")

            calls: list[tuple[list[str], dict[str, object]]] = []

            def original_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                calls.append((list(cmd), dict(kwargs)))
                self.assertEqual(cmd[0], str(base), "the broken managed interpreter must not seed the child venv")
                if "-c" in cmd:
                    self.assertTrue(stale_marker.exists(), "the target must survive until the seed is validated")
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        stdout=seed_probe_payload(base, base.parent),
                        stderr="",
                    )
                if "--clear" in cmd and target.exists():
                    shutil.rmtree(target)
                scripts = target / "Scripts"
                scripts.mkdir(parents=True)
                (scripts / "python.exe").touch()
                return subprocess.CompletedProcess(cmd, 0)

            # Modly v0.4 replaces subprocess.run with this transparent wrapper
            # before executing extension setup.py through runpy.run_path().
            def _patched_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                forwarded = list(args)
                return original_run(forwarded[0], **kwargs)

            source_env = contaminated_env()
            with (
                mock.patch.object(setup_module.sys, "executable", str(managed)),
                mock.patch.object(setup_module.sys, "_base_executable", str(base), create=True),
                mock.patch.object(setup_module.subprocess, "run", side_effect=_patched_run),
            ):
                setup_module.create_extension_venv(str(managed), target, env=source_env)

            self.assertEqual(len(calls), 2)
            probe_command, probe_kwargs = calls[0]
            self.assertEqual(probe_command[:2], [str(base), "-E"])
            self.assertIn("-c", probe_command)
            self.assertTrue(probe_kwargs["check"])
            self.assertTrue(probe_kwargs["capture_output"])
            self.assertTrue(probe_kwargs["text"])
            probe_env = probe_kwargs["env"]
            self.assertIsInstance(probe_env, dict)
            self.assertTrue(REDIRECTORS.isdisjoint(probe_env))
            self.assertEqual(probe_env["PATH"], source_env["PATH"])
            self.assertEqual(probe_env["CUDA_PATH"], source_env["CUDA_PATH"])

            command, kwargs = calls[1]
            self.assertEqual(
                command,
                [str(base), "-E", "-m", "venv", "--clear", str(target)],
            )
            child_env = kwargs["env"]
            self.assertIsInstance(child_env, dict)
            self.assertTrue(REDIRECTORS.isdisjoint(child_env))
            self.assertEqual(child_env["PATH"], source_env["PATH"])
            self.assertEqual(child_env["CUDA_PATH"], source_env["CUDA_PATH"])
            self.assertFalse(stale_marker.exists())
            self.assertTrue((target / "Scripts" / "python.exe").exists())

    def test_rejects_unusable_foreign_or_incoherent_seed_before_clearing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            managed = root / "dependencies" / "venv" / "Scripts" / "python.exe"
            base = root / "python-base" / "python.exe"
            target = root / "extensions" / "trellis-text" / "venv"
            managed.parent.mkdir(parents=True)
            base.parent.mkdir(parents=True)
            target.mkdir(parents=True)
            managed.touch()
            base.touch()
            marker = target / "installed-environment.txt"

            failures = {
                "unusable": subprocess.CalledProcessError(1, [str(base), "-E", "-c", "probe"]),
                "foreign-version": seed_probe_payload(base, base.parent, version=(3, 10)),
                "incoherent-prefix": seed_probe_payload(
                    base,
                    base.parent,
                    base_prefix=root / "different-python-base",
                ),
            }
            for name, outcome in failures.items():
                with self.subTest(name=name):
                    marker.write_text("keep", encoding="utf-8")

                    def reject_probe(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                        self.assertIn("-c", cmd, "venv --clear must not run before validation succeeds")
                        if isinstance(outcome, BaseException):
                            raise outcome
                        return subprocess.CompletedProcess(cmd, 0, stdout=outcome, stderr="")

                    with (
                        mock.patch.object(setup_module.sys, "executable", str(managed)),
                        mock.patch.object(setup_module.sys, "_base_executable", str(base), create=True),
                        mock.patch.object(setup_module.subprocess, "run", side_effect=reject_probe),
                    ):
                        with self.assertRaisesRegex(RuntimeError, "Python seed"):
                            setup_module.create_extension_venv(str(managed), target, env=contaminated_env())

                    self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


class RealNestedVenvSmokeTests(unittest.TestCase):
    def test_managed_venv_process_recreates_stale_extension_venv_offline(self) -> None:
        driver = """
import importlib.util
import sys
from pathlib import Path

setup_path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("trellis_text_nested_smoke", setup_path)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load setup.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.create_extension_venv(sys.executable, Path(sys.argv[2]))
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "paths with spaces"
            outer = root / "dependencies" / "venv"
            inner = root / "extensions" / "trellis-text" / "venv"
            inner.mkdir(parents=True)
            stale_marker = inner / "partial-ensurepip-install.txt"
            stale_marker.write_text("stale", encoding="utf-8")

            subprocess.run(
                [sys.executable, "-E", "-m", "venv", "--clear", str(outer)],
                check=True,
                capture_output=True,
                text=True,
            )
            managed_python = venv_python(outer)
            nested_setup = subprocess.run(
                [managed_python, "-E", "-c", driver, str(ROOT / "setup.py"), str(inner)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(nested_setup.returncode, 0, nested_setup.stderr)

            extension_python = venv_python(inner)
            pip_probe = subprocess.run(
                [extension_python, "-E", "-m", "pip", "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(stale_marker.exists())
            self.assertTrue(extension_python.is_file())
            self.assertIn("pip ", pip_probe.stdout)


if __name__ == "__main__":
    unittest.main()
