"""Export the FastAPI contract without connecting to deployment services."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Destination JSON file")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    # Import in a clean process/directory so even invalid local settings cannot affect the contract.
    with tempfile.TemporaryDirectory(prefix="sivera-openapi-") as directory:
        result = subprocess.run(
            [sys.executable, "-c", "import json; from api.main import app; print(json.dumps(app.openapi(), ensure_ascii=False, indent=2))"],
            cwd=directory, env={"PYTHONPATH": str(root)},
            capture_output=True, text=True, encoding="utf-8", check=True,
        )
    args.output.write_text(result.stdout, encoding="utf-8")


if __name__ == "__main__":
    main()
