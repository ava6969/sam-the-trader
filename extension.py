
import pandas as pd
from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities

# Set the start and end dates of the bars, should also align with the Trading Calendar
start_session = pd.Timestamp('2003-1-3', tz='utc')
end_session = pd.Timestamp('2020-12-30', tz='utc')


register(
    'custom-bundle',   # What to call the new bundle
    csvdir_equities(
        ['minute'],  # Are these daily or minute bars
        '/home/dewe/sam/datasets/',  # Directory where the formatted bar data is
    ),
    calendar_name='NYSE', # US equities default
    start_session=start_session,
    end_session=end_session
)
