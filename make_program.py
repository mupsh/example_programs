import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
CONSTANTS
"""

# TODO: AT 60s used and 30fps - 0.02 freq resolution


FREQUENCY = [0.1,0.2,0.4]
for i in range(len(FREQUENCY)):
    TMAX = 80
    ELECTRODE_MAX_R = 75000
    ELECTRODE_MIN_R = 250
    DT = 0.1
    FILENAME = "bob"

    """
    GENERATOR
    """
    t = np.arange(0, TMAX, DT)

    ramp_up_time = 100  # sec

    # ramp up left side stimuli to avoid abrupt stimulus
    ramp_factor = (1 / ramp_up_time) * t
    ramp_factor[t > ramp_up_time] = 1
    ramp_factor[t > TMAX - ramp_up_time] = (-1 / ramp_up_time) * t[t > TMAX - ramp_up_time] + (TMAX / ramp_up_time)

    signal = ((ELECTRODE_MAX_R + ELECTRODE_MIN_R) / 2 * np.sin(2 * np.pi * FREQUENCY[i] * t) + (
                ELECTRODE_MAX_R + ELECTRODE_MIN_R) / 2)*ramp_factor
    signal_opposite = ((ELECTRODE_MAX_R + ELECTRODE_MIN_R) / 2 * -np.sin(2 * np.pi * FREQUENCY[i] * t) + (
                ELECTRODE_MAX_R + ELECTRODE_MIN_R) / 2)*ramp_factor
    plt.plot(t, signal)
    #plt.figure()
    plt.plot(t, signal_opposite)
    #plt.figure()

"""
MAKE PROGRAM
"""

program = np.zeros((len(t), 11))
program[:, 0] = t
program[:, 1] = signal
program[:, 2] = signal_opposite
program[:, 3:] = ELECTRODE_MAX_R

program = np.asarray(program, dtype='int')

"""
SAVE PROGRAM
"""

program = pd.DataFrame(program, columns=None)
program[0] = t
program.to_csv(FILENAME + ".csv", header=None, index=None)