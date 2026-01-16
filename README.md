# SHIELD (Smoke & Heat Indoor Exposure Limiting Decision-tool)

SHIELD is a Windows desktop decision tool that predicts **indoor wildfire smoke exposure (PM2.5)** and **indoor heat risk** for the next **72 hours**, then generates an optimized hour-by-hour action plan (“SHIELD Mode”) for homes, classrooms, and clinics.

## What SHIELD does
- Forecasts **indoor PM2.5** and **indoor temperature / heat stress** with uncertainty bands
- Produces an optimized hourly plan: windows/ventilation timing, purifier/fan schedule, and relocation triggers
- Works in **Offline Demo Mode** (no internet needed) using redistributable CC-BY datasets and cached scenarios
- Respects data licensing (OpenAQ license-aware; PurpleAir raw data not redistributed)

## Quick start (dev)
> These commands assume you are on Windows PowerShell and have a conda environment.

```powershell
conda create -n shield python=3.11 -y
conda activate shield
pip install -U pip
pip install -e .
python -m shield.app