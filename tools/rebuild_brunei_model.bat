@echo off
setlocal

cd /d D:\Brunei

echo ========================================
echo Brunei ROI DINO/PatchCore Full Rebuild
echo ========================================
echo.

if not exist "configs\brunei.json" (
  echo ERROR: configs\brunei.json not found.
  echo Run: git pull --ff-only origin main
  exit /b 1
)

if not exist "dataset_roi_dino\train\good" (
  echo ERROR: dataset_roi_dino\train\good not found.
  exit /b 1
)

if not exist "artifacts\reference\reference_aligned.png" (
  echo ERROR: artifacts\reference\reference_aligned.png not found.
  exit /b 1
)

if not exist "D:\wlenai\third_party\dinov2\hubconf.py" (
  echo ERROR: DINOv2 repo not found at D:\wlenai\third_party\dinov2
  exit /b 1
)

if not exist "D:\wlenai\weights\dinov2_vits14_pretrain.pth" (
  echo ERROR: DINOv2 weights not found.
  exit /b 1
)

echo Removing old model directory...
if exist "artifacts\roi_dino_full" rmdir /s /q "artifacts\roi_dino_full"

echo.
echo Building fresh 19-ROI model from GOOD images...
python tools\build_roi_dino_patchcore.py ^
  --good-dir "D:\Brunei\dataset_roi_dino\train\good" ^
  --reference "D:\Brunei\artifacts\reference\reference_aligned.png" ^
  --config "D:\Brunei\configs\brunei.json" ^
  --output "D:\Brunei\artifacts\roi_dino_full" ^
  --dino-repo "D:\wlenai\third_party\dinov2" ^
  --dino-weights "D:\wlenai\weights\dinov2_vits14_pretrain.pth" ^
  --device auto ^
  --image-size 224 ^
  --coreset 0.10 ^
  --coreset-projection-dim 64 ^
  --calibration-ratio 0.20 ^
  --threshold-margin 1.10 ^
  --score-top-fraction 0.05 ^
  --seed 42 ^
  --threshold 238 ^
  --min-inlier-ratio 0.25

if errorlevel 1 (
  echo.
  echo REBUILD FAILED.
  exit /b 1
)

echo.
echo ========================================
echo REBUILD COMPLETE
echo ========================================
echo Model: D:\Brunei\artifacts\roi_dino_full\model.json
echo Banks: D:\Brunei\artifacts\roi_dino_full\banks
echo.
dir "D:\Brunei\artifacts\roi_dino_full\banks"

endlocal
