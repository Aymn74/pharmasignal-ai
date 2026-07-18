@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto :install

where py >nul 2>nul
if errorlevel 1 goto :create_with_python
py -3.12 -m venv .venv
goto :venv_created

:create_with_python
python -m venv .venv

:venv_created
if errorlevel 1 goto :error

:install

".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :error

start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://localhost:8501'"
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.address localhost
goto :eof

:error
echo.
echo PharmaSignal AI could not start. Review the error above.
pause
exit /b 1
