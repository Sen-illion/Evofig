<#
Deploy one local commit to an isolated checkout on the training server.

The server keeps its own working copy at <DATA_ROOT>/EvoFig, and a paid
batch can be importing that tree lazily for hours. Replacing files under a
running shard can hand a half-written module to a process that has not reached
its next import yet, so this script never writes to it: it bundles the requested
local commit, uploads the bundle, and brings a *second* checkout up to that
commit. The running batch keeps the code it started with; newer work gets a tree
of its own.

The bundle is incremental when the server copy already carries the base commit
this workstation branched from, and the whole history otherwise, so a first
deploy is the only one that moves tens of megabytes.

Usage (from anywhere in the repository):

  pwsh scripts/deploy_isolated_checkout.ps1 `
    -Target <DATA_ROOT>/EvoFig-se `
    -KeyPath "<PATH>"

Pass -SkipChecks to land the commit without running the far-side compile and
self-check; the checks are what makes a silent half-deploy visible, so they are
on by default. Pass -BaseRef to state the base explicitly instead of reading it
off the server checkout.
#>
[CmdletBinding()]
param(
    [string]$SshHost = "<HOST>",
    [int]$Port = 52743,
    [string]$User = "root",
    [string]$KeyPath = (Join-Path $env:USERPROFILE ".ssh\<SSH_KEY>"),
    [string]$ServerRepo = "<DATA_ROOT>/EvoFig",
    [string]$Target = "<DATA_ROOT>/EvoFig-se",
    [string]$Python = "<DATA_ROOT>/envs/paperimage/bin/python",
    [string]$Ref = "HEAD",
    [string]$BaseRef = "",
    [string]$BranchName = "se",
    [string]$Git = "<PATH>",
    [switch]$SkipChecks
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Git)) {
    $Git = "git"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$suffix = [guid]::NewGuid().ToString("N").Substring(0, 8)
$bundlePath = Join-Path $env:TEMP "_deploy_$suffix.bundle"
$remoteBundle = "<DATA_ROOT>/_deploy_$suffix.bundle"

function Invoke-Remote {
    param([Parameter(Mandatory = $true)][string]$Script, [switch]$AllowFailure)
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Script))
    $output = & ssh -i $KeyPath -p $Port -o StrictHostKeyChecking=no "$User@$SshHost" `
        "echo $encoded | base64 -d > <DATA_ROOT>/_deploy_step.sh; bash <DATA_ROOT>/_deploy_step.sh" 2>&1
    $code = $LASTEXITCODE
    $text = ($output | ForEach-Object { "$_" }) -join "`n"
    if ($code -ne 0 -and -not $AllowFailure) {
        throw "remote step exited with $code`n$text"
    }
    return [pscustomobject]@{ ExitCode = $code; Text = $text }
}

$commit = (& $Git -C $repoRoot rev-parse --verify "$Ref^{commit}").Trim()
$shortCommit = $commit.Substring(0, 7)
Write-Host "deploying $commit to ${Target} on $SshHost"

if (-not $BaseRef) {
    # The base is only a hint that makes the upload small; a checkout that cannot
    # be read is not a reason to stop, it just means the full history goes up.
    $probe = Invoke-Remote -AllowFailure -Script "cd $ServerRepo 2>/dev/null && git rev-parse HEAD"
    if ($probe.ExitCode -eq 0) { $BaseRef = $probe.Text.Trim() }
}

if ($BaseRef) {
    & $Git -C $repoRoot bundle create $bundlePath $Ref --not $BaseRef 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "base $BaseRef does not reach $Ref; uploading the full history instead"
        $BaseRef = ""
    }
}
if (-not $BaseRef) {
    & $Git -C $repoRoot bundle create $bundlePath $Ref 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "git bundle create failed" }
}
& $Git bundle verify $bundlePath 2>&1 | Out-Null
$size = (Get-Item -LiteralPath $bundlePath).Length
Write-Host ("uploading {0:N0} bytes" -f $size)

& scp -i $KeyPath -P $Port -o StrictHostKeyChecking=no $bundlePath "$User@${SshHost}:$remoteBundle" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { throw "scp failed" }

# The remote step is assembled from single quoted templates so that the shell
# substitutions reach bash instead of being expanded here.
$checkTemplate = @'
__PYTHON__ -m compileall -q rfs && echo "compileall ok"
__PYTHON__ -c "import rfs; print('imported', rfs.__file__)"
__PYTHON__ -m rfs.cli doctor --json > /dev/null && echo "doctor ok"
'@
$checks = if ($SkipChecks) { "echo 'checks skipped'" } else { $checkTemplate.Replace('__PYTHON__', $Python) }

$landTemplate = @'
set -e
if [ ! -d __TARGET__/.git ]; then git clone -q --local __SERVER_REPO__ __TARGET__; fi
cd __TARGET__
git fetch -q __BUNDLE__ HEAD
git checkout -q -B __BRANCH__ FETCH_HEAD
echo "landed on $(git log --oneline -1)"
echo "dirty files: $(git status --short | wc -l)"
__CHECKS__
'@
$land = $landTemplate.
    Replace('__TARGET__', $Target).
    Replace('__SERVER_REPO__', $ServerRepo).
    Replace('__BUNDLE__', $remoteBundle).
    Replace('__BRANCH__', $BranchName).
    Replace('__CHECKS__', $checks)

try {
    $result = Invoke-Remote -Script $land
    Write-Host $result.Text
    if ($result.Text -notmatch "landed on $shortCommit") {
        throw "the checkout did not report the commit that was sent ($commit)"
    }
} finally {
    Invoke-Remote -AllowFailure -Script "rm -f $remoteBundle" | Out-Null
    # The bundle is scratch, and on Windows a just-read temporary file can still
    # be mapped for a moment: never let that turn a good deploy into a failure.
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
        try {
            Remove-Item -LiteralPath $bundlePath -Force -ErrorAction Stop
            break
        } catch {
            Start-Sleep -Milliseconds 400
        }
    }
}

Write-Host ""
Write-Host "run this checkout's code with:"
Write-Host "  ssh -i `"$KeyPath`" -p $Port $User@$SshHost `"cd $Target && $Python -m rfs.cli <command>`""
