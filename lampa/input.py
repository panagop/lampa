from dataclasses import dataclass
from dataclasses_json import dataclass_json
from functools import cached_property
import numpy as np
import pandas as pd
import pystrata

# import json
# from enum import Enum, auto
# import pyexcel


@dataclass_json
@dataclass
class LTimeSeriesMotion:
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
    def from_txt(filename: str, description: str = "", skiprows: int = 2) -> "LTimeSeriesMotion":
        df = pd.read_csv(filename, header=None, skiprows=skiprows,
                         encoding="utf-8", delim_whitespace=True)
        return LTimeSeriesMotion(
            description=description,
            time_step=np.round(df[0][1] - df[0][0], 6),  # df[0].diff().mean(),
            accels=df[1].to_numpy()
        )

    @staticmethod
    def from_csv(filename: str, description: str = "", delimiter: str = ",", skiprows: int = 1) -> "LTimeSeriesMotion":
        df = pd.read_csv(filename, header=None, skiprows=skiprows,
                         encoding="utf-8", delimiter=delimiter)
        return LTimeSeriesMotion(
            description=description,
            time_step=df[0][1] - df[0][0],  # df[0].diff().mean(),
            accels=df[1].to_numpy()
        )

    @staticmethod
    def from_excel(filename: str, sheet_name: str | int = 0, description: str = "", skiprows: int = 1) -> "LTimeSeriesMotion":
        df = pd.read_excel(filename, sheet_name=sheet_name,
                           header=None, skiprows=skiprows)
        return LTimeSeriesMotion(
            description=description,
            time_step=df[0][1] - df[0][0],  # df[0].diff().mean(),
            accels=df[1].to_numpy()
        )


@dataclass_json
@dataclass
class LSoilType:
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
class LDarendeliSoilType:
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
class LLayer:
    layer_type: str
    layer_properties: LSoilType | LDarendeliSoilType
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
class LPyStrataInput:
    name: str
    calculator_type: str  # LCalculatorType
    time_series_motion: LTimeSeriesMotion
    layers: list[LLayer]

    @cached_property
    def to_pystrata_profile(self) -> pystrata.site.Profile:
        """Convert to pystrata profile"""
        return pystrata.site.Profile([layer.to_pystrata for layer in self.layers]).auto_discretize()

    # def save_json_file(self, filename: str):
    #     """Save to json file"""
    #     with open(filename, "w") as f:
    #         f.write(self.to_json())

    # @staticmethod
    # def from_json_file(filename: str) -> "LPyStrataInput":
    #     """Load from json file"""
    #     with open(filename, "r") as f:
    #         return LPyStrataInput.from_json(f.read())

    def save_json_file(self, filename: str):
        """Save to json file"""
        with open(filename, "w") as f:
            f.write(LPyStrataInput.schema().dumps(self))

    @staticmethod
    def from_json_file(filename: str) -> "LPyStrataInput":
        """Load from json file"""
        with open(filename, "r") as f:
            return LPyStrataInput.schema().loads(f.read())