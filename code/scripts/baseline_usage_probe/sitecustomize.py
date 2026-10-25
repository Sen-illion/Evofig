"""Install the baseline usage probe in any interpreter that has this directory on PYTHONPATH.

``site`` imports ``sitecustomize`` by name at startup, which is how a baseline subprocess can be
measured without editing a single line of the vendored system it runs.
"""

try:
    from usage_probe import install
except Exception:  # pragma: no cover - the probe must never block an interpreter start
    install = None

if install is not None:
    try:
        install()
    except Exception:  # pragma: no cover - same reason
        pass
