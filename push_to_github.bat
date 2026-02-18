@echo off
echo ========================================
echo Hand Gesture - Android APK Build Helper
echo ========================================
echo.

REM Check if git is initialized
if not exist ".git" (
    echo [1/4] Initializing Git repository...
    git init
    echo.
)

REM Add all files
echo [2/4] Adding files to Git...
git add .
git status
echo.

REM Commit
set commit_msg=Update for Android build
set /p commit_msg="Enter commit message (default: Update for Android build): "
git commit -m "%commit_msg%"
echo.

REM Check if remote exists
git remote -v >nul 2>&1
if errorlevel 1 (
    echo.
    echo ========================================
    echo Please create a GitHub repository first, then run:
    echo   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
    echo   git branch -M main
    echo   git push -u origin main
    echo ========================================
    echo.
    echo Then GitHub Actions will automatically build the APK!
) else (
    echo [3/4] Pushing to GitHub...
    git branch -M main
    git push -u origin main
    echo.
    echo [4/4] Done!
    echo.
    echo ========================================
    echo Please visit GitHub repository Actions page
    echo to check build progress
    echo After build completes, download APK from Artifacts
    echo ========================================
)

echo.
pause
