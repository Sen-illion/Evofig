param(
    [string]$TestPath = "tests"
)

$ErrorActionPreference = "Stop"
$tmpRoot = Join-Path (Get-Location) ".test-tmp"
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null
$env:TEMP = $tmpRoot
$env:TMP = $tmpRoot

try {
    python -m unittest discover -s $TestPath -q
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    if (Test-Path -LiteralPath $tmpRoot) {
        Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
