@echo off
setlocal
cd /d "%~dp0"

for /L %%i in (1,1,5) do (
    call :train_round %%i
)

echo.
echo All 5 training attempts completed.
pause
exit /b 0

:train_round
echo.
echo ===== Training attempt %1 / 5 =====
if exist "models\c1a0_ppo_checkpoint_d_reward_v2_continued.zip" goto train_resume
if exist "models\c1a0_ppo_checkpoint_d_reward_v2.zip" goto train_resume
goto train_fresh

:train_resume
echo Existing model found. Continuing training from saved data.
python.exe main.py --train --resume --timesteps 500 --model-name c1a0_ppo_checkpoint_d_reward_v2
goto report_result

:train_fresh
echo No saved model found. Starting training from scratch.
python.exe main.py --train --timesteps 500 --model-name c1a0_ppo_checkpoint_d_reward_v2

:report_result
if errorlevel 1 (
    echo Training attempt %1 exited with an error. Continuing to the next attempt.
) else (
    echo Training attempt %1 finished normally.
)
exit /b 0
