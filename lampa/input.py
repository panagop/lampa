﻿import json
from dataclasses import dataclass
from enum import Enum, auto
# from fileinput import filename

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyexcel
import pystrata
from dataclasses_json import dataclass_json
from pystrata.site import Profile

import streamlit as st


@dataclass_json
@dataclass
class TimeSeriesMotion:
    description: str
    time_step: float
    accels: np.array

    @property
    def to_pystrata(self) -> pystrata.motion.TimeSeriesMotion:
        return pystrata.motion.TimeSeriesMotion(
            filename=None,
            description=self.description,
            time_step=self.time_step,
            accels=self.accels,
        )

    @staticmethod
    def from_txt(filename: str, description: str = "", skiprows: int = 2) -> "TimeSeriesMotion":
        df = pd.read_csv(filename, header=None, skiprows=skiprows,
                         encoding="utf-8", delim_whitespace=True)
        return TimeSeriesMotion(
            description=description,
            time_step=np.round(df[0][1] - df[0][0], 6),  # df[0].diff().mean(),
            accels=df[1].to_numpy()
        )

    @staticmethod
    def from_csv(filename: str, description: str = "", delimiter: str = ",", skiprows: int = 1) -> "TimeSeriesMotion":
        df = pd.read_csv(filename, header=None, skiprows=skiprows,
                         encoding="utf-8", delimiter=delimiter)
        return TimeSeriesMotion(
            description=description,
            time_step=df[0][1] - df[0][0],  # df[0].diff().mean(),
            accels=df[1].to_numpy()
        )

    @staticmethod
    def from_excel(filename: str, sheet_name: str | int = 0, description: str = "", skiprows: int = 1) -> "TimeSeriesMotion":
        df = pd.read_excel(filename, sheet_name=sheet_name,
                           header=None, skiprows=skiprows)
        return TimeSeriesMotion(
            description=description,
            time_step=df[0][1] - df[0][0],  # df[0].diff().mean(),
            accels=df[1].to_numpy()
        )


@dataclass_json
@dataclass
class SoilType:
    name: str
    unit_wt: float
    damping: float = 0.05

    @property
    def to_pystrata(self) -> pystrata.site.SoilType:
        """Convert to pystrata soil type"""
        return pystrata.site.SoilType(name=self.name,
                                      unit_wt=self.unit_wt,
                                      mod_reduc=None,
                                      damping=self.damping)


@dataclass_json
@dataclass
class DarendeliSoilType:
    name: str
    unit_wt: float
    plas_index: float = 0.0
    ocr: float = 1.0
    stress_mean: float = 101.3
    freq: float = 1.0
    num_cycles: float = 10.0

    @property
    def to_pystrata(self) -> pystrata.site.DarendeliSoilType:
        """Convert to pystrata Darendeli soil type"""
        return pystrata.site.DarendeliSoilType(unit_wt=self.unit_wt,
                                               plas_index=self.plas_index,
                                               ocr=self.ocr,
                                               stress_mean=self.stress_mean,
                                               freq=self.freq,
                                               num_cycles=self.num_cycles)


@dataclass_json
@dataclass
class Layer:
    layer_type: str
    layer_properties: SoilType | DarendeliSoilType
    thickness: float
    shear_vel: float

    @property
    def to_pystrata(self) -> pystrata.site.Layer:
        """Convert to pystrata layer"""
        return pystrata.site.Layer(self.layer_properties.to_pystrata,
                                   thickness=self.thickness,
                                   shear_vel=self.shear_vel)


@dataclass_json
@dataclass
class PyStrataInput:
    name: str
    calculator_type: str
    time_series_motion: TimeSeriesMotion
    layers: list[Layer]

    @property
    def to_pystrata_profile(self) -> pystrata.site.Profile:
        """Convert to pystrata profile"""
        return pystrata.site.Profile([layer.to_pystrata for layer in self.layers]).auto_discretize()


# class CalculatorType(Enum):
#     LINEAR_ELASTIC_CALCULATOR = auto()
#     EQUIVALENT_LINEAR_CALCULATOR = auto()


# @dataclass
# class Input:
#     time_series_motion: pystrata.motion.TimeSeriesMotion
#     site_layers: list[pystrata.site.Layer]
#     calculator_type: CalculatorType

#     strain_limit: float = 0.05

#     def do_the_calcs(self, strain_limit: float = 0.05):
#         profile = pystrata.site.Profile(self.site_layers).auto_discretize()

#         if self.calculator_type == CalculatorType.LINEAR_ELASTIC_CALCULATOR:
#             calculator = pystrata.propagation.LinearElasticCalculator()
#         elif self.calculator_type == CalculatorType.EQUIVALENT_LINEAR_CALCULATOR:
#             calculator = pystrata.propagation.EquivalentLinearCalculator(
#                 self.strain_limit)
