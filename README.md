# pywumpuslite

A light-weight WUMPUS environment, written in Python.

The original version of this project is (functionally) the same as James P. Biagioni's Wumpus-Lite. The only difference is that this project is written in Python, rather than Java.

## Setup

```sh
git clone git@github.com:jackblandin/pywumpuslite.git && cd pywumpuslite
```

```sh
pip install -r requirements.txt
```

## Run via command line

```sh
python world_application.py -t 100
```

#### Optional arguments

* `-d` <int> world size
* `-s` <int> max steps
* `-t` <int> number of trials
* `-a` <bool> random agent location
* `-r` <int> user-specified seed
* `-f` <str> output filename
* `-n` <bool> non-deterministic world

## Run in Jupyter notebook or other python file

```py
import world_application

world_application.run_application(num_trials=100)
```
