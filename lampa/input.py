from dataclasses import dataclass
from dataclasses_json import dataclass_json

from fileinput import filename
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pystrata
import pyexcel
import json

from pystrata.motion import TimeSeriesMotion
from pystrata.site import Profile

@dataclass_json
@dataclass
class PyStrataInput:
    name: str
    time_series_motion: dict[str, np.array]

