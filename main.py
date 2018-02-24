import argparse
import json
import multiprocessing as mp
import os
from time import time

from pympler.asizeof import asizeof

from data_structure import FullyDynClus
from utils import data_stream


def fully_dynamic(eps, args):
    """
    Run the fully dynamic clustering algorithm with the given parameter eps.
    """
    print('Running the algorithm with eps = {} in Process #{}'.format(
        eps, os.getpid()
    ))

    limit = args.limit
    window = args.window

    space = 0  # keep track of the space used by the data structures
    start = time()
    beta = 0

    fdc = FullyDynClus(eps, 20, window)

    for point in data_stream(limit):
        # There are some duplicate points in the dataset, for example,
        # the 71st and 86th points
        if point in fdc.points:
            continue

        fdc.insert(point)

        # Delete the least recent point in the sliding window
        if len(fdc.points) >= window + 1:
            fdc.delete(fdc.points[-window - 1])

        beta = max(beta, fdc.get_result())
        space = max(space, asizeof(fdc))

    print('Finish running the algorithm with eps = {}'.format(eps))

    return {str(eps): {'run_time': round(time() - start, 3),
                       'space': space,
                       'op_count': fdc.op_count,
                       'beta': beta}}


def check_args(parser, args):
    if args.limit < args.window:
        parser.error('The number of points needs to be at least the window size.')

    return args


def parse_args():
    parser = argparse.ArgumentParser(description='Fully Dynamic k-center Clustering algorithm.')
    parser.add_argument('-c', '--cpu', type=int, default=os.cpu_count(),
                        help="The number of parallel processes to run the algorithm. "
                             "Default: os.cpu_count() = %(default)s.")
    parser.add_argument('-l', '--limit', type=int, default=1000,
                        help="The total number of points to be inserted in the algorithm. "
                             "Default: %(default)s.")
    parser.add_argument('-w', '--window', type=int, default=60,
                        help="The size of the sliding window. Default: %(default)s.")

    args = parser.parse_args()
    args = check_args(parser, args)

    return args


def main():
    args = parse_args()
    results = {}

    def collect_result(result):
        """
        Collect and update the result of an instance into the overall results
        """
        results.update(result)

    # Create the pool of processes
    pool = mp.Pool(args.cpu)

    for eps in [i / 10 for i in range(1, 11)]:
        pool.apply_async(fully_dynamic, args=(eps, args), callback=collect_result)

    # Start running the algorithm
    pool.close()
    pool.join()

    with open('results.json', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
