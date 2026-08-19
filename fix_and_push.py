import os
import shutil
import subprocess
from pathlib import Path

# Repository Configuration
REPO_URL = "https://github.com/Abhinavyukth-Suresh/NeuralPy.git"
ROOT_DIR = Path(__file__).parent.resolve()
SRC_PACKAGE_DIR = ROOT_DIR / "src" / "neuralpy"
TESTS_DIR = ROOT_DIR / "tests"

# File migration map (converts duplicates and upper-case filenames)
FILE_MAPPING = {
    "Activations.py": "activations.py",
    "Activation_functions.py": "activations.py",
    "Dense.py": "layers.py",
    "layer.py": "layers.py",
    "Layers.py": "layers.py",
    "ERROR.py": "errors.py",
    "errors.py": "errors.py",
    "MLP.py": "mlp.py",
    "NN.py": "nn.py",
    "optimizers.py": "optimizers.py",
    "__init__.py": "__init__.py"
}

PYPROJECT_CONTENT = """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "neuralpy"
version = "0.1.0"
description = "A lightweight neural network library built from scratch in Python"
readme = "README.md"
requires-python = ">=3.8"
license = { file = "LICENSE" }
dependencies = [
    "numpy>=1.20.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
]

[build-system.targets.wheel]
packages = ["src/neuralpy"]
"""

GITIGNORE_CONTENT = """# Python artifacts
__pycache__/
*.py[cod]
*$py.class
*.so

# Environments
.venv/
env/

# Build & Distribution
build/
dist/
*.egg-info/
.eggs/

# IDE
.vscode/
.idea/
"""

def run_cmd(cmd, check=True):
    """Executes shell commands and prints real-time output."""
    print(f" Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True, cwd=ROOT_DIR)
    if res.stdout.strip():
        print(f"    {res.stdout.strip()}")
    if res.stderr.strip() and res.returncode != 0:
        print(f"  ⚠️ Error/Warning: {res.stderr.strip()}")
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return res

def clean_and_restructure():
    print("\n--- 1. Restructuring Local Directory ---")
    
    # Create directories
    SRC_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    (TESTS_DIR / "__init__.py").touch(exist_ok=True)

    # Move and consolidate files
    for old_filename, new_filename in FILE_MAPPING.items():
        old_path = ROOT_DIR / old_filename
        target_path = SRC_PACKAGE_DIR / new_filename

        if old_path.exists():
            print(f"  -> Migrating {old_filename} to src/neuralpy/{new_filename}")
            content = old_path.read_text(encoding="utf-8", errors="ignore")

            if target_path.exists() and old_filename != new_filename:
                with open(target_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n# --- Merged from {old_filename} ---\n\n")
                    f.write(content)
            else:
                target_path.write_text(content, encoding="utf-8")

            old_path.unlink()

    # Generate pyproject.toml & .gitignore
    (ROOT_DIR / "pyproject.toml").write_text(PYPROJECT_CONTENT, encoding="utf-8")
    (ROOT_DIR / "gitignore").rename(ROOT_DIR / ".gitignore") if (ROOT_DIR / "gitignore").exists() else None
    (ROOT_DIR / ".gitignore").write_text(GITIGNORE_CONTENT, encoding="utf-8")

def reset_and_push_git():
    print("\n--- 2. Resetting Git and Pushing to GitHub ---")

    git_dir = ROOT_DIR / ".git"
    if git_dir.exists():
        print("  -> Removing existing corrupt/stale .git folder...")
        if os.name == "nt":
            subprocess.run(f'rmdir /s /q "{git_dir}"', shell=True)
        else:
            shutil.rmtree(git_dir)

    # Initialize Git fresh
    run_cmd("git init")
    run_cmd("git branch -M main")
    run_cmd(f"git remote add origin {REPO_URL}")

    # Stage, commit, and push
    run_cmd("git add -A")
    run_cmd('git commit -m "refactor: restructure package to src-layout and standardize module names"')
    
    print("\n--- Pushing to GitHub ---")
    run_cmd("git push -u origin main --force")

if __name__ == "__main__":
    try:
        clean_and_restructure()
        reset_and_push_git()
        
        # Self-delete the setup script when done
        script_path = Path(__file__)
        if script_path.exists():
            script_path.unlink()

        print("\n SUCCESS! Your repository has been completely restructured and pushed to GitHub.")
        print(f"Check your repo here: https://github.com/Abhinavyukth-Suresh/NeuralPy")

    except Exception as e:
        print(f"\n❌ Execution failed: {e}")