# Zorya Dynamics TDA repo

- run venv on `python3.12` due to the dependencies
- activate venv and install deps from `requirements.txt` via `pip install -r requirements.txt` command

## MLFlow & Optuna setup

I followed this [yt vid](https://www.youtube.com/watch?v=H4Fd7wsueZw):

1. install and use `uv` package manager, it's the way to go
1. run `uv venv --python 3.12` because tensorflow doesn't work yet in 3.13
1. remember about using python env
1. install deps `uv pip install --requirements requirements.txt`
1. run in this dir `uv pip install optuna mlflow` (yes, I should have updated the reqs file)
1. Run in a separate terminal session: `uv run mlflow server --host 0.0.0.0 --port 8080 --backend-store-uri sqlite:///my.db` .It will 
1. You can access ML flow dashboard in a browser: http://localhost:8080 . Data is stored in file `my.db` in a dir where the terminal was.
1. Run in a separate terminal session: `uv run cal_and_test_optuna.py` . That's it!
