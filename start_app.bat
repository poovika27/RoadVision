@echo off
REM Starts the pothole website on the GPU (WSL) -> open http://127.0.0.1:5000
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/c/Users/mourish/Desktop/Mine/pothole-cnn && ~/tfgpu/bin/python app.py"
