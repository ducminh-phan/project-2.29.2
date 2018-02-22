import json
import multiprocessing as mp
import os
from time import time

from pympler.asizeof import asizeof

from data_structure import FullyDynClus
from utils import data_stream

MODE = 'SMALL'
CONFIG = {
    'FULL': (1000000, 60000),
    'SMALL': (1000, 60)
}
LIMIT, WINDOW = CONFIG[MODE]


def fully_dynamic(eps):
    """
    Run the fully dynamic clustering algorithm with the given parameter eps.
    """
    print('Running the algorithm with ε = {} in Process #{}'.format(
        eps, os.getpid()
    ))

    space = 0  # keep track of the space used by the data structures
    start = time()

    fdc = FullyDynClus(eps, 20, WINDOW)

    for point in data_stream(LIMIT):
        # There are some duplicate points in the dataset, for example,
        # the 71st and 86th points
        if point in fdc.points:
            continue

        fdc.insert(point)

        # Delete the least recent point in the sliding window
        if len(fdc.points) >= WINDOW + 1:
            fdc.delete(fdc.points[-WINDOW - 1])

        space = max(space, asizeof(fdc))

        # if len(fdc.points) % 100 == 0:
        #     print(len(fdc.points) // 100, int(time() - start))

    print('Finish running the algorithm with ε = {}'.format(eps))

    return {str(eps): {'run_time': round(time() - start, 3),
                       'space': space,
                       'op_count': fdc.op_count}}


def main():
    results = {}

    def collect_result(result):
        """
        Collect and update the result of an instance into the overall results
        """
        results.update(result)

    # Create the pool of processes
    pool = mp.Pool()

    for eps in [i / 10 for i in range(1, 11)]:
        pool.apply_async(fully_dynamic, args=(eps,), callback=collect_result)

    # Starting running the algorithm
    pool.close()
    pool.join()

    with open('results.json', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
