# Expose a class NPRAlaska which provides DataLoader instances for bundled LAS files.
# This version is resilient to running as a package or running from source (no installed package).
from pathlib import Path

try:
    # stdlib (py3.7+)
    from importlib import resources
except Exception:
    # backport when supporting older Python versions and you've added importlib_resources
    import importlib_resources as resources  # type: ignore

# Import DataLoader: prefer relative import when used as a package, otherwise use absolute.
try:
    from ..data_management.preprocessing import DataLoader
except Exception:
    from stoneforge.data_management.preprocessing import DataLoader  # type: ignore

__all__ = ["NPRAlaska"]


class NPRAlaska:
    """
    Container for bundled datasets in this package.

    Usage:
        from stoneforge import datasets
        npr = datasets.NPRAlaska()
        dp1_loader = npr.dp1
    """

    def __init__(self):
        # instantiate each DataLoader using a helper that prefers importlib.resources
        self.dp1 = self._load_resource("DP1.las")
        self.es1 = self._load_resource("ES1.las")
        self.ik1 = self._load_resource("IK1.las")
        self.wd1 = self._load_resource("WD1.las")

    def _load_resource(self, name: str) -> "DataLoader":
        """
        Return a DataLoader constructed from a packaged resource named `name`.

        Strategy:
        - When this module is imported as a package (__package__ truthy), try to use
          importlib.resources.path(...) to get a usable filesystem path (works for
          normal installs and editable installs; for zipped packages it may extract to a
          temporary file).
        - If that fails (or __package__ is falsy because code is executed directly),
          fall back to a path relative to this file on the filesystem.
        """
        # Try using importlib.resources when we're a real package.
        if __package__:
            try:
                with resources.path(__package__, name) as p:
                    # Construct the DataLoader while inside the context manager so that
                    # any temporary extraction done by importlib.resources remains available.
                    return DataLoader(str(p), filetype="las2")
            except Exception:
                # fall back to filesystem path below
                pass

        # Fallback: resource is expected to live next to this file in the source tree.
        p = Path(__file__).resolve().parent / name
        return DataLoader(str(p), filetype="las2")