@echo off
setlocal
cd /d "%~dp0"

for /L %%i in (1,1,20) do (
    call :train_round %%i
)

echo.
echo All 20 training rounds completed.
pause
exit /b 0

:train_round
echo.
echo ===== Training round %1 / 20 =====
if exist "models\c1a0_ppo_all_binds_continued.zip" goto train_resume
if exist "models\c1a0_ppo_all_binds.zip" goto train_resume
goto train_fresh

:train_resume
echo Existing model found. Continuing training from saved data.
python.exe main.py --train --resume --timesteps 3000
goto report_result

:train_fresh
echo No saved model found. Starting training from scratch.
python.exe main.py --train --timesteps 3000

:report_result
if errorlevel 1 (
    echo Training round %1 exited with an error. Continuing to the next round.
) else (
    echo Training round %1 finished normally.
)
exit /b 0
