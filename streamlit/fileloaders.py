import pandas as pd
from pystrata.motion import TimeSeriesMotion


def time_accel_txt_to_pystrata_motion(filename: str) -> TimeSeriesMotion:
    df = pd.read_csv(filename, header=None, skiprows=2,
                     encoding="utf-8", delim_whitespace=True)
    accels = df[1].to_numpy()
    time_step = df[0][1] - df[0][0]

    return TimeSeriesMotion(filename=filename, description='', time_step=time_step, accels=accels)


def time_accel_excel_to_pystrata_motion(filename: str) -> TimeSeriesMotion:
    # not ready yet
    df = pd.read_excel(filename, header=None, skiprows=2,
                       encoding="utf-8", delim_whitespace=True)
    accels = df[1].to_numpy()
    time_step = df[0][1] - df[0][0]

    return TimeSeriesMotion(filename=filename, description='', time_step=time_step, accels=accels)


def time_accel_at2_to_pystrata_motion(filename: str) -> TimeSeriesMotion:
    # not ready yet
    df = pd.read_excel(filename, header=None, skiprows=2,
                       encoding="utf-8", delim_whitespace=True)
    accels = df[1].to_numpy()
    time_step = df[0][1] - df[0][0]

    return TimeSeriesMotion(filename=filename, description='', time_step=time_step, accels=accels)
