@echo off
echo [MUG] Setting up environment...
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
echo [MUG] Done! Run 'run.py' to start.
pause