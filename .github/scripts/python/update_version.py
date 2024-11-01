"""
This CLI script is used to update the version of the package. It is used by the
CI/CD pipeline to update the version of the package when a new release is made.

It uses argparse to parse the command line arguments, which are the new version
and the path to the package's __init__.py file.
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Update the version of the package."
    )
    parser.add_argument(
        "--version",
        type=str,
        help="The new version of the package.",
        required=True,
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="The path to the package's version file.",
    )
    args = parser.parse_args()

    with open(args.path, "w") as f:
        f.write(f"__version__ = \"{args.version}\"")
        

if __name__ == "__main__":
    main()
