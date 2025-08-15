# dukascopy_downloader.py

from datetime import datetime, timedelta
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_GBP_USD

start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)
instrument = INSTRUMENT_FX_MAJORS_GBP_USD
interval = dukascopy_python.INTERVAL_MIN_1
offer_side = dukascopy_python.OFFER_SIDE_BID

df = dukascopy_python.fetch(
    instrument,
    interval,
    offer_side,
    start,
    end,
)


df.to_csv("./data/2024min1.csv")

