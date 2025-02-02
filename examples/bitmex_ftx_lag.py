import logging
import sys
from datetime import datetime
from multiprocessing import cpu_count
from os import PathLike
from pathlib import Path

import pandas as pd

from lead_lag import LeadLag


def parse_ftx_date(s: str) -> datetime:
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f%z")
    except ValueError:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")
    raise ValueError(f"Unknown date format: {s}")


def main():
    bitmex_file = Path('../data/XBTUSD.csv.zip')
    ftx_file = Path('../data/BTC-PERP.csv.zip')
    bitmex, ftx = read_data(bitmex_file, ftx_file)

    ll = LeadLag(ts1=ftx, ts2=bitmex, max_lag=1, verbose=False, min_precision=0.001)
    print('Running inference...')
    ll.run_inference(num_threads=cpu_count() // 2)
    print(f'Estimated lag: {ll.lead_lag} seconds.')
    print(f'Positive lag means ts1 is leading. LLR: {ll.llr:.2f} (cf. paper for the definition of LLR).')
    ll.plot_results()


def read_data(bitmex_file: PathLike | str, ftx_file: PathLike | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    bitmex = pd.read_csv(bitmex_file, index_col="timestamp", parse_dates=True, date_format="%Y-%m-%dD%H:%M:%S.%f000", compression='zip')
    bitmex = bitmex[bitmex['symbol'] == 'XBTUSD']
    bitmex = bitmex[bitmex['price'].diff() != 0]
    bitmex = bitmex['price']
    ftx = pd.read_csv(ftx_file, index_col="time", compression='zip')
    ftx = ftx[ftx['price'].diff() != 0]
    ftx = ftx['price']
    ftx.index = ftx.index.map(parse_ftx_date)
    return bitmex, ftx


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
    main()
