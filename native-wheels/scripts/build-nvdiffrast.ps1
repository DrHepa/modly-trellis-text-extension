[CmdletBinding()]
param(
    [string]$Python = "py -3.11",
    [string]$OutDir = ".\native-wheels\wheelhouse",
    [string]$WorkDir = ".\native-wheels\work\nvdiffrast",
    [string]$TorchVersion = "2.7.0",
    [string]$TorchVisionVersion = "0.22.0",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [string]$CudaRoot = $env:CUDA_PATH,
    [string]$TorchCudaArchList = "8.6;8.9;9.0+PTX",
    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoUrl = 'https://github.com/NVlabs/nvdiffrast.git'
$RepoRef = 'v0.4.0'

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

function Ensure-Directory {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    New-Item -ItemType Directory -Force -Path $PathValue | Out-Null
}

$resolvedOutDir = [System.IO.Path]::GetFullPath($OutDir)
$resolvedWorkDir = [System.IO.Path]::GetFullPath($WorkDir)
$venvDir = Join-Path $resolvedWorkDir 'venv'
$srcDir = Join-Path $resolvedWorkDir 'src'
$repoDir = Join-Path $srcDir 'nvdiffrast'

if ($Clean -and (Test-Path $resolvedWorkDir)) {
    Remove-Item -Recurse -Force $resolvedWorkDir
}

Ensure-Directory -PathValue $resolvedOutDir
Ensure-Directory -PathValue $srcDir

if (-not $CudaRoot) {
    throw 'CUDA Toolkit root was not resolved. Pass -CudaRoot or set CUDA_PATH/CUDA_HOME.'
}

$env:CUDA_HOME = $CudaRoot
$env:CUDA_PATH = $CudaRoot
$env:CUDACXX = (Join-Path $CudaRoot 'bin\nvcc.exe')
$env:TORCH_CUDA_ARCH_LIST = $TorchCudaArchList
$env:DISTUTILS_USE_SDK = '1'

Invoke-Python -Arguments @('-m', 'venv', '"' + $venvDir + '"')
$venvPython = Join-Path $venvDir 'Scripts\python.exe'

& $venvPython -m pip install --upgrade pip setuptools wheel ninja
& $venvPython -m pip install "torch==$TorchVersion" "torchvision==$TorchVisionVersion" --index-url $TorchIndexUrl

if (Test-Path $repoDir) {
    Remove-Item -Recurse -Force $repoDir
}

git clone $RepoUrl "$repoDir"
git -C "$repoDir" checkout $RepoRef

& $venvPython -m pip wheel "$repoDir" --no-build-isolation -w "$resolvedOutDir"

Write-Host "[native-wheels] nvdiffrast wheel build complete: $resolvedOutDir"
