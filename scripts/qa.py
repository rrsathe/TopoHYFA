#!/usr/bin/env python
"""
Unified QA and Development Workflow Automation Script for TopoHYFA.
Runs linting, formatting, type checking, tests, coverage, dependency audits,
and Docker validations in a cross-platform manner.
"""

import argparse
import subprocess
import sys


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> bool:
    """Run a system command and return True if successful (exit code 0), False otherwise."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    try:
        res = subprocess.run(cmd, check=False, env=env)
        if res.returncode != 0:
            print(f"[FAIL] Command failed with exit code: {res.returncode}", flush=True)
            return False
        print("[OK] Command succeeded", flush=True)
        return True
    except Exception as e:
        print(f"[ERROR] Exception running command: {e}", flush=True)
        return False


def lint(fix: bool = False) -> bool:
    print("\n=== Running Ruff Linter ===")
    cmd = ["uv", "run", "ruff", "check"]
    if fix:
        cmd.append("--fix")
    cmd.append(".")
    return run_command(cmd)


def format_code() -> bool:
    print("\n=== Running Ruff Formatter ===")
    return run_command(["uv", "run", "ruff", "format", "."])


def typecheck() -> bool:
    print("\n=== Running Static Type Checking (ty) ===")
    return run_command(["uv", "run", "ty", "check"])


def run_tests(cov: bool = False) -> bool:
    print("\n=== Running Tests ===")
    cmd = ["uv", "run", "pytest"]
    if cov:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-branch"])
    return run_command(cmd)


def security_audit() -> bool:
    print("\n=== Running Security Audit (pip-audit) ===")
    return run_command(["uv", "run", "pip-audit"])


def dependency_audit() -> bool:
    print("\n=== Running Dependency Audit (deptry) ===")
    return run_command(["uv", "run", "deptry", "."])


def docker_build() -> bool:
    print("\n=== Building Production Docker Image ===")
    return run_command(["docker", "build", "-t", "topohyfa:latest", "."])


def docker_test() -> bool:
    print("\n=== Validating Docker Compose & Smoke Testing ===")

    compose_ok = run_command(["docker", "compose", "config"])
    if not compose_ok:
        return False

    smoke_cmd = [
        "docker",
        "run",
        "--rm",
        "topohyfa:latest",
        "python",
        "-c",
        "import src.data; print('[OK] TopoHYFA Python imports verified in container')",
    ]
    return run_command(smoke_cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="TopoHYFA QA and Task Runner Tool")
    parser.add_argument(
        "action",
        choices=[
            "lint",
            "format",
            "typecheck",
            "test",
            "cov",
            "security",
            "deptry",
            "docker-build",
            "docker-test",
            "qa",
        ],
        help="The automation workflow action to execute.",
    )
    parser.add_argument(
        "--fix", action="store_true", help="Apply safe automatic fixes during linting"
    )
    args = parser.parse_args()

    success = True

    if args.action == "lint":
        success = lint(args.fix)
    elif args.action == "format":
        success = format_code()
    elif args.action == "typecheck":
        success = typecheck()
    elif args.action == "test":
        success = run_tests(cov=False)
    elif args.action == "cov":
        success = run_tests(cov=True)
    elif args.action == "security":
        success = security_audit()
    elif args.action == "deptry":
        success = dependency_audit()
    elif args.action == "docker-build":
        success = docker_build()
    elif args.action == "docker-test":
        success = docker_test()
    elif args.action == "qa":
        print("\n=======================================================")
        print("=== RUNNING FULL TOPOHYFA QUALITY ASSURANCE PIPELINE ===")
        print("=======================================================")

        steps = [
            ("Format", format_code),
            ("Lint", lambda: lint(fix=True)),
            ("Type Check", typecheck),
            ("Tests & Coverage", lambda: run_tests(cov=True)),
            ("Dependency Audit", dependency_audit),
            ("Security Audit", security_audit),
            ("Docker Build", docker_build),
            ("Docker Test & Smoke", docker_test),
        ]

        failures = []
        for name, func in steps:
            print(f"\n--- STEP: {name} ---")
            step_success = func()
            if not step_success:
                failures.append(name)
                print(f"[FAIL] Step '{name}' FAILED!")
            else:
                print(f"[OK] Step '{name}' PASSED.")

        print("\n=======================================================")
        if failures:
            print(f"[FAIL] QA PIPELINE FAILED! Failed steps: {', '.join(failures)}")
            success = False
        else:
            print("[OK] QA PIPELINE SUCCEEDED! All checks passed.")
            success = True

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
