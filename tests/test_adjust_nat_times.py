from pathlib import Path
import sys

import pandas as pd
import pytz

sys.path.append(str(Path(__file__).resolve().parents[1]))

from trade_analysis_script import adjust_nat_times


def test_adjust_nat_times_uses_matching_symbol_fallbacks():
    tz = pytz.timezone('US/Eastern')

    data = {
        'Symbol': ['ES', 'NQ', 'ES', 'CL', 'NQ', 'ES', 'CL'],
        'OpenClose': ['open', 'open', 'close', 'close', 'close', 'close', 'close'],
        'TransDateTime': [
            pd.Timestamp('2024-01-01 09:00', tz=tz),
            pd.Timestamp('2024-01-01 09:02', tz=tz),
            pd.NaT,
            pd.NaT,
            pd.NaT,
            pd.Timestamp('2024-01-01 09:10', tz=tz),
            pd.Timestamp('2024-01-01 09:20', tz=tz),
        ],
    }

    df = pd.DataFrame(data)

    adjusted_df = adjust_nat_times(df.copy())

    expected_es_close = pd.Timestamp('2024-01-01 09:01', tz=tz)
    expected_nq_close = pd.Timestamp('2024-01-01 09:03', tz=tz)
    expected_cl_close = pd.Timestamp('2024-01-01 09:19', tz=tz)

    assert adjusted_df.loc[2, 'TransDateTime'] == expected_es_close
    assert adjusted_df.loc[4, 'TransDateTime'] == expected_nq_close
    assert adjusted_df.loc[3, 'TransDateTime'] == expected_cl_close

    # Ensure existing timestamps remain unchanged
    assert adjusted_df.loc[5, 'TransDateTime'] == df.loc[5, 'TransDateTime']
    assert adjusted_df.loc[6, 'TransDateTime'] == df.loc[6, 'TransDateTime']
