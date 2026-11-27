@echo off
cd /d "%~dp0"
echo ======================================================
echo Pushing to https://github.com/y1697342661-creator/AI
echo If a GitHub login window pops up, click "Sign in with your browser".
echo ======================================================
echo.
"C:\Program Files\Git\cmd\git.exe" push -u origin main
echo.
echo Done! If you see "branch 'main' set up to track 'origin/main'", push succeeded.
pause
