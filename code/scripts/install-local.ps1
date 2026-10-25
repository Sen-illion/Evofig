$ErrorActionPreference = "Stop"
python -m pip install -e <PATH>
Write-Host "Installed rfs console script. Test with: rfs doctor --json"
