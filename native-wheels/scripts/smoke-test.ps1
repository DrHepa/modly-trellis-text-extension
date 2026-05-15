[CmdletBinding()]
param(
    [string]$Python = "py -3.11",
    [string]$WheelDir = ".\native-wheels\wheelhouse",
    [string]$TorchVersion = "2.7.0",
    [string]$TorchVisionVersion = "0.22.0",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [switch]$KeepVenv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )

    $command = "$Python $($Arguments -join ' ')"
    Write-Host "[native-wheels] $command"
    if ($WorkingDirectory) {
        Push-Location $WorkingDirectory
        try {
            Invoke-Expression $command
        }
        finally {
            Pop-Location
        }
        return
    }
    Invoke-Expression $command
}

$resolvedWheelDir = [System.IO.Path]::GetFullPath($WheelDir)
if (-not (Test-Path $resolvedWheelDir)) {
    throw "Wheel directory not found: $resolvedWheelDir"
}

$nvdiffrastWheel = Get-ChildItem -Path $resolvedWheelDir -Filter 'nvdiffrast-*.whl' | Select-Object -First 1
$diffGaussianWheel = Get-ChildItem -Path $resolvedWheelDir -Filter 'diff_gaussian_rasterization-*.whl' | Select-Object -First 1

if (-not $nvdiffrastWheel) {
    throw 'No nvdiffrast wheel found in wheelhouse.'
}
if (-not $diffGaussianWheel) {
    throw 'No diff_gaussian_rasterization wheel found in wheelhouse.'
}

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('modly-trellis-text-native-smoke-' + [System.Guid]::NewGuid().ToString('N'))
$venvDir = Join-Path $tempRoot 'venv'

Invoke-Python -Arguments @('-m', 'venv', '"' + $venvDir + '"')
$venvPython = Join-Path $venvDir 'Scripts\python.exe'

try {
    & $venvPython -m pip install --upgrade pip setuptools wheel
    & $venvPython -m pip install "torch==$TorchVersion" "torchvision==$TorchVisionVersion" --index-url $TorchIndexUrl
    & $venvPython -m pip install --no-deps "$($nvdiffrastWheel.FullName)" "$($diffGaussianWheel.FullName)"
    & $venvPython -c "import torch; import nvdiffrast.torch; import diff_gaussian_rasterization; print('native smoke OK', torch.__version__)"
    Write-Host '[native-wheels] Smoke test passed.'
}
finally {
    if (-not $KeepVenv -and (Test-Path $tempRoot)) {
        Remove-Item -Recurse -Force $tempRoot
    }
}
