from dataclasses import dataclass
from functools import cached_property
import numpy as np
import pandas as pd
import pystrata
from lampa.input import LPyStrataInput as LInput

FREQS = np.logspace(-0.5, 2, num=500)
# FREQS = np.logspace(-1, 2, num=500)

@dataclass
class LProject:
    l_imput: LInput

    @cached_property
    def calculator(self):
        # Create the profile
        profile = self.l_imput.to_pystrata_profile

        # LCalculatorType.LINEAR_ELASTIC_CALCULATOR:
        if self.l_imput.calculator_type == 'LinearElasticCalculator':
            calc = pystrata.propagation.LinearElasticCalculator()
        # LCalculatorType.EQUIVALENT_LINEAR_CALCULATOR:
        elif self.l_imput.calculator_type == 'EquivalentLinearCalculator':
            calc = pystrata.propagation.EquivalentLinearCalculator()

        calc(self.l_imput.time_series_motion.to_pystrata, profile,
             profile.location("outcrop", index=-1))

        return calc

    def response_spectrum(self,
                          freqs: np.ndarray = FREQS,
                          damping: float = 0.05,
                          location_index=0) -> pystrata.output.ResponseSpectrumOutput:
        """Response spectrum"""
        out = pystrata.output.ResponseSpectrumOutput(
            # Frequency
            freqs,
            # Location of the output
            pystrata.output.OutputLocation("outcrop", index=location_index),
            # Damping
            damping,
        )

        out(self.calculator)
        return out

    def accel_transfer_function(self,
                                freqs: np.ndarray = FREQS) -> pystrata.output.AccelTransferFunctionOutput:
        """Acceleration transfer function"""
        out = pystrata.output.AccelTransferFunctionOutput(
            # Frequency
            freqs,
            # Location in (denominator),
            pystrata.output.OutputLocation("outcrop", index=-1),
            # Location out (numerator)
            pystrata.output.OutputLocation("outcrop", index=0),
        )

        out(self.calculator)
        return out

    def response_spectrum_ratio(self,
                            freqs: np.ndarray = FREQS,
                            damping: float = 0.05) -> pystrata.output.ResponseSpectrumRatioOutput:
        out = pystrata.output.ResponseSpectrumRatioOutput(
             # Frequency
            freqs,
            # Location in (denominator),
            pystrata.output.OutputLocation("outcrop", index=-1),
            # Location out (numerator)
            pystrata.output.OutputLocation("outcrop", index=0),
            # Damping
            damping,
        )           

        out(self.calculator)
        return out


    def fourier_amplitude_spectrum(self,
                                   freqs: np.ndarray = FREQS,
                                   ko_bandwidth:int=30) -> pystrata.output.FourierAmplitudeSpectrumOutput:
        """Fourier amplitude spectrum"""
        out = pystrata.output.FourierAmplitudeSpectrumOutput(
            # Frequency
            freqs,
            # Location of the output
            pystrata.output.OutputLocation("outcrop", index=0),
            # Bandwidth for Konno-Omachi smoothing window
            ko_bandwidth=ko_bandwidth,
        )

        out(self.calculator)
        return out  