# Project MPRI 2.29.2

This repository is for the project of the course MPRI 2.29.2: Graph Mining. The goal is to implement the Fully Dynamic k-center Clustering algorithm.

## Setup

This implementation was tested on Python 3.5, but it should be able to run on Python 3.4+ without any problem. Additional packages can be installed using the command `pip3 install -r requirements.txt`.

## Running

The script can be run via CLI: `python3 main.py [OPTIONS]`

### Options

```
  -h, --help            show this help message and exit
  -p PROC, --proc PROC  The number of parallel processes to run the algorithm.
                        Default: os.cpu_count() = 8.
  -l LIMIT, --limit LIMIT
                        The total number of points to be inserted in the
                        algorithm. Default: 1000.
  -w WINDOW, --window WINDOW
                        The size of the sliding window. Default: 60.

```

### Analysis

The results will be saved in the file `results_{limit}_{window}.json`. The relevant plots can be obtained from the notebook `report.ipynb`. This notebook uses the `seaborn` library to obtain cleaner plots, but it is easy to remove the usage of `seaborn` if necessary.
