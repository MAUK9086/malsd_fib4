@echo off
REM Run any project script with PYTHONPATH set to the project root.
REM Usage: run.bat src/define_cohort.py
REM        run.bat src/exp03_feature_engineering.py

set PYTHONPATH=%~dp0
python %*
