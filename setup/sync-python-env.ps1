$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "[sync] Running uv sync..." -ForegroundColor Cyan
uv sync --preview-features extra-build-dependencies

Write-Host "[sync] Verifying mmcv native ops..." -ForegroundColor Cyan
$check = & .\.venv\Scripts\python.exe -c "import mmcv._ext; from mmcv.ops import nms; print('MMCV native ops: OK')" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host $check
    Write-Host "[sync] Environment looks good." -ForegroundColor Green
    exit 0
}

Write-Warning "mmcv native ops check failed after plain uv sync. Reinstalling mmcv from the pinned OpenMMLab index..."
Write-Host $check

uv sync --preview-features extra-build-dependencies --reinstall-package mmcv --link-mode=copy

Write-Host "[sync] Re-checking mmcv native ops..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -c "import mmcv._ext; from mmcv.ops import nms; print('MMCV native ops: OK')"

if ($LASTEXITCODE -ne 0) {
    throw "mmcv native ops still failed after reinstall."
}

Write-Host "[sync] Environment repaired successfully." -ForegroundColor Green
