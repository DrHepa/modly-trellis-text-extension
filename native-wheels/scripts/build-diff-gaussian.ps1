[CmdletBinding()]
param(
    [string]$Python = "py -3.11",
    [string]$OutDir = ".\native-wheels\wheelhouse",
    [string]$WorkDir = ".\native-wheels\work\diff-gaussian",
    [string]$TorchVersion = "2.7.0",
    [string]$TorchVisionVersion = "0.22.0",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [string]$CudaRoot = $env:CUDA_PATH,
    [string]$TorchCudaArchList = "6.1;7.5;8.0;8.6;8.9;9.0+PTX",
    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoUrl = 'https://github.com/autonomousvision/mip-splatting.git'
$RepoRef = 'dda02ab5ecf45d6edb8c540d9bb65c7e451345a9'
$PackageSubdir = 'submodules/diff-gaussian-rasterization'

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )

    $pythonParts = $Python -split ' '
    $pythonExe = $pythonParts[0]
    $pythonArgs = @()
    if ($pythonParts.Length -gt 1) {
        $pythonArgs = $pythonParts[1..($pythonParts.Length - 1)]
    }
    $allArgs = @($pythonArgs) + @($Arguments)
    Write-Host "[native-wheels] $pythonExe $($allArgs -join ' ')"
    if ($WorkingDirectory) {
        Push-Location $WorkingDirectory
        try {
            & $pythonExe @allArgs
        }
        finally {
            Pop-Location
        }
        return
    }
    & $pythonExe @allArgs
}

function Ensure-Directory {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    New-Item -ItemType Directory -Force -Path $PathValue | Out-Null
}

function Ensure-CudaCcclHeaders {
    param([Parameter(Mandatory = $true)][string]$CudaRoot)

    $cudaInclude = Join-Path $CudaRoot 'include'
    $cudaNvInclude = Join-Path $cudaInclude 'nv'
    $cudaNvTarget = Join-Path $cudaNvInclude 'target'
    if (Test-Path $cudaNvTarget) {
        Write-Host "[native-wheels] CUDA CCCL nv/target header already available at: $cudaNvTarget"
        return
    }

    $candidateNvIncludes = @(
        (Join-Path $CudaRoot 'include\cccl\nv'),
        (Join-Path $CudaRoot 'targets\x86_64-windows\include\nv'),
        (Join-Path $CudaRoot 'targets\x86_64-win32\include\nv')
    )
    $sourceNvInclude = $candidateNvIncludes | Where-Object { Test-Path (Join-Path $_ 'target') } | Select-Object -First 1

    if (-not $sourceNvInclude) {
        $foundNvTarget = Get-ChildItem -Path $CudaRoot -Recurse -File -Filter target -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '[\\/]nv[\\/]target$' } |
            Select-Object -First 1
        if ($foundNvTarget) {
            $sourceNvInclude = Split-Path $foundNvTarget.FullName -Parent
        }
    }

    if (-not $sourceNvInclude) {
        throw "Could not locate CUDA CCCL nv/target header under $CudaRoot. Install the NVIDIA Conda cuda-cccl_win-64 package or provide a full CUDA Toolkit layout."
    }

    Ensure-Directory -PathValue $cudaNvInclude
    Copy-Item -Path (Join-Path $sourceNvInclude '*') -Destination $cudaNvInclude -Recurse -Force
    Write-Host "[native-wheels] Mirrored CUDA CCCL nv headers from $sourceNvInclude to: $cudaNvInclude"
}

function Ensure-CudaWindowsLibLayout {
    param([Parameter(Mandatory = $true)][string]$CudaRoot)

    $cudaLib = Join-Path $CudaRoot 'lib'
    $cudaLibX64 = Join-Path $cudaLib 'x64'
    Ensure-Directory -PathValue $cudaLibX64

    if (Test-Path $cudaLib) {
        Copy-Item -Path (Join-Path $cudaLib '*.lib') -Destination $cudaLibX64 -Force -ErrorAction SilentlyContinue
    }

    $cudartLib = Join-Path $cudaLibX64 'cudart.lib'
    if (-not (Test-Path $cudartLib)) {
        throw "Could not locate cudart.lib at $cudartLib. The Windows CUDA library layout is incomplete for PyTorch extension linking."
    }

    $env:LIB = "$cudaLibX64;$cudaLib;$env:LIB"
    Write-Host "[native-wheels] CUDA Windows library layout ready: $cudaLibX64"
}

$resolvedOutDir = [System.IO.Path]::GetFullPath($OutDir)
$resolvedWorkDir = [System.IO.Path]::GetFullPath($WorkDir)
$venvDir = Join-Path $resolvedWorkDir 'venv'
$srcDir = Join-Path $resolvedWorkDir 'src'
$repoDir = Join-Path $srcDir 'mip-splatting'
$packageDir = Join-Path $repoDir $PackageSubdir

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
Write-Host "[native-wheels] TORCH_CUDA_ARCH_LIST=$env:TORCH_CUDA_ARCH_LIST"

$ccclInclude = Join-Path $CudaRoot 'include\cccl'
if (Test-Path $ccclInclude) {
    $env:INCLUDE = "$ccclInclude;$env:INCLUDE"
    Write-Host "[native-wheels] Added CUDA CCCL include path: $ccclInclude"
}
Ensure-CudaCcclHeaders -CudaRoot $CudaRoot
Ensure-CudaWindowsLibLayout -CudaRoot $CudaRoot

Invoke-Python -Arguments @('-m', 'venv', $venvDir)
$venvPython = Join-Path $venvDir 'Scripts\python.exe'

& $venvPython -m pip install --upgrade pip setuptools wheel ninja
& $venvPython -m pip install "torch==$TorchVersion" "torchvision==$TorchVisionVersion" --index-url $TorchIndexUrl

if (Test-Path $repoDir) {
    Remove-Item -Recurse -Force $repoDir
}

git clone $RepoUrl "$repoDir"
git -C "$repoDir" checkout $RepoRef
git -C "$repoDir" submodule update --init --recursive

if (-not (Test-Path (Join-Path $packageDir 'third_party\glm'))) {
    throw 'GLM submodule was not populated under diff-gaussian-rasterization/third_party/glm.'
}

& $venvPython -m pip wheel "$packageDir" --no-build-isolation -w "$resolvedOutDir"

Write-Host "[native-wheels] diff_gaussian_rasterization wheel build complete: $resolvedOutDir"
