import numpy as np
import pandas as pd
import portion as P


NANOSECONDS_IN_MILLISECOND = 1e6
NANOSECONDS_IN_SECOND = NANOSECONDS_IN_MILLISECOND * 1000
NANOSECONDS_IN_MINUTE = NANOSECONDS_IN_SECOND * 60
NANOSECONDS_IN_HOUR = NANOSECONDS_IN_MINUTE * 60

def convert_time_series_to_events(vector):
    """
    Convert time series (a list of timestamps and values > 0 indicating for the anomalous instances)
    to a list of events. The events are considered as durations,
    i.e. setting 1 at index i corresponds to an anomalous interval [i, i+1).

    :param vector: a list of elements belonging to {0, 1}
    :return: a list of couples, each couple representing the start and stop of
    each event
    """
    vector = np.asarray(vector)

    def find_runs(x):
        """Find runs of consecutive items in an array."""

        # ensure array
        x = np.asanyarray(x)
        if x.ndim != 1:
            raise ValueError('only 1D array supported')
        n = x.shape[0]

        # handle empty array
        if n == 0:
            return np.array([]), np.array([]), np.array([])

        else:
            # find run starts
            loc_run_start = np.empty(n, dtype=bool)
            loc_run_start[0] = True
            np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
            run_starts = np.nonzero(loc_run_start)[0]

            # find run values
            run_values = x[loc_run_start]

            # find run lengths
            run_lengths = np.diff(np.append(run_starts, n))

            run_ends = run_starts + run_lengths

            return np.stack((run_starts[run_values > 0], run_ends[run_values > 0])).transpose()

    non_zero_runs = find_runs(vector[..., 1])

    events = []
    n = len(vector)
    for x, y in non_zero_runs:
        if y == n:
            events.append(P.closed(vector[..., 0][x], vector[..., 0][y - 1]))
        else:
            events.append(P.closedopen(vector[..., 0][x], vector[..., 0][y]))
    events = P.Interval(*events)

    return events


def label_changes(timestamps, labels) -> list[tuple[pd.Timestamp, int]]:
    """``(timestamp, label)`` pairs of a binary series reduced to its first row, the rows
    where the label changes and its last row.

    ``convert_time_series_to_events`` makes each run of positives an event from the run's
    first row to the row after it (or to the last row), and the metrics built on it (ADTQC,
    channel- and subsystem-aware scores) otherwise read only the first and last
    timestamps, so the reduced series gives the same results with memory proportional to
    the label changes rather than to the samples. A series whose timestamps are not sorted
    is returned whole.
    """
    ts = pd.DatetimeIndex(timestamps)
    labels = np.asarray(labels)
    if len(labels) == 0 or not ts.is_monotonic_increasing:
        return list(zip(ts, labels.tolist()))
    keep = np.empty(len(labels), bool)
    keep[0] = keep[-1] = True
    np.not_equal(labels[1:-1], labels[:-2], out=keep[1:-1])
    idx = np.flatnonzero(keep)
    return list(zip(ts[idx], labels[idx].tolist()))
