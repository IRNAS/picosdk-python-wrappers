#
# Copyright (C) 2018 Pico Technology Ltd. See LICENSE file for terms.
#
# PS2000 BLOCK MODE EXAMPLE
# This example opens a 2000 driver device, sets up two channels and a trigger then collects a block of data.
# This data is then plotted as mV against time in ns.

import ctypes
import numpy as np
from picosdk.ps2000 import ps2000 as ps
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico2000_ok
import time
import csv
from scipy.optimize import leastsq

def fit_sin_to_data(timebase,data, guess_phase = 0, guess_freq = 1, guess_amp = 1):

    N = 1000 # number of data points
    t = timebase # nanoseconds
    guess_mean = np.mean(data)
    guess_std = 3*np.std(data)/(2**0.5)/(2**0.5)

    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = guess_std*np.sin((t+guess_phase)*guess_freq) + guess_mean

    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) + x[3] - data
    est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]

    # recreate the fitted curve using the optimized parameters
    data_fit = est_amp*np.sin(est_freq*t+est_phase) + est_mean

    # recreate the fitted curve using the optimized parameters

    fine_t = timebase
    data_fit=est_amp*np.sin(est_freq*fine_t+est_phase)+est_mean

    return data_fit

# Create status ready for use
status = {}

# Open 2000 series PicoScope
# Returns handle to chandle for use in future API functions
status["openUnit"] = ps.ps2000_open_unit()
assert_pico2000_ok(status["openUnit"])

# Create chandle for use
chandle = ctypes.c_int16(status["openUnit"])
try:
    while True:
        # Set up channel A
        # handle = chandle
        # channel = PS2000_CHANNEL_A = 0
        # enabled = 1
        # coupling type = PS2000_DC = 1 CHANGED TO AC
        # range = PS2000_2V = 7
        # analogue offset = 0 V
        chARange = 8
        status["setChA"] = ps.ps2000_set_channel(chandle, 0, 1, 0, chARange)
        assert_pico2000_ok(status["setChA"])

        # Set up channel B
        # handle = chandle
        # channel = PS2000_CHANNEL_B = 1
        # enabled = 1
        # coupling type = PS2000_DC = 1
        # range = PS2000_2V = 7
        # analogue offset = 0 V
        chBRange = 7
        status["setChB"] = ps.ps2000_set_channel(chandle, 1, 1, 1, chBRange)
        assert_pico2000_ok(status["setChB"])

        # Set up single trigger
        # handle = chandle
        # source = PS2000_CHANNEL_A = 0
        # threshold = 1024 ADC counts
        # direction = PS2000_RISING = 0
        # delay = 0 s
        # auto Trigger = 1000 ms
        status["trigger"] = ps.ps2000_set_trigger(chandle, 0, 64, 0, 0, 1000)
        assert_pico2000_ok(status["trigger"])

        # Set number of pre and post trigger samples to be collected
        preTriggerSamples = 1000
        postTriggerSamples = 1000
        maxSamples = preTriggerSamples + postTriggerSamples

        # Get timebase information
        # handle = chandle
        # timebase = 8 = timebase
        # no_of_samples = maxSamples
        # pointer to time_interval = ctypes.byref(timeInterval)
        # pointer to time_units = ctypes.byref(timeUnits)
        # oversample = 1 = oversample
        # pointer to max_samples = ctypes.byref(maxSamplesReturn)
        timebase = 13
        timeInterval = ctypes.c_int32()
        timeUnits = ctypes.c_int32()
        oversample = ctypes.c_int16(1)
        maxSamplesReturn = ctypes.c_int32()
        status["getTimebase"] = ps.ps2000_get_timebase(chandle, timebase, maxSamples, ctypes.byref(timeInterval), ctypes.byref(timeUnits), oversample, ctypes.byref(maxSamplesReturn))
        assert_pico2000_ok(status["getTimebase"])


        # Run block capture
        # handle = chandle
        # no_of_samples = maxSamples
        # timebase = timebase
        # oversample = oversample
        # pointer to time_indisposed_ms = ctypes.byref(timeIndisposedms)
        timeIndisposedms = ctypes.c_int32()
        status["runBlock"] = ps.ps2000_run_block(chandle, maxSamples, timebase, oversample, ctypes.byref(timeIndisposedms))
        assert_pico2000_ok(status["runBlock"])

        # Check for data collection to finish using ps5000aIsReady
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            status["isReady"] = ps.ps2000_ready(chandle)
            ready = ctypes.c_int16(status["isReady"])

        # Create buffers ready for data
        bufferA = (ctypes.c_int16 * maxSamples)()
        bufferB = (ctypes.c_int16 * maxSamples)()

        # Get data from scope
        # handle = chandle
        # pointer to buffer_a = ctypes.byref(bufferA)
        # pointer to buffer_b = ctypes.byref(bufferB)
        # poiner to overflow = ctypes.byref(oversample)
        # no_of_values = cmaxSamples
        cmaxSamples = ctypes.c_int32(maxSamples)
        status["getValues"] = ps.ps2000_get_values(chandle, ctypes.byref(bufferA), ctypes.byref(bufferB), None, None, ctypes.byref(oversample), cmaxSamples)
        assert_pico2000_ok(status["getValues"])

        # find maximum ADC count value
        maxADC = ctypes.c_int16(32767)

        # convert ADC counts data to mV
        adc2mVChA =  adc2mV(bufferA, chARange, maxADC)
        adc2mVChB =  adc2mV(bufferB, chBRange, maxADC)

        # Create time data
        timebase = np.linspace(0, (cmaxSamples.value) * timeInterval.value, cmaxSamples.value)
        fit = fit_sin_to_data(timebase, adc2mVChA, guess_freq = 0.00000031, guess_amp = 2300)
        
        idx = np.argmax(fit)

        if idx is not None:
            print("idx = %d"%(idx))
            valB = adc2mVChB[:][idx]
            valA = adc2mVChA[:][idx]
            print(idx, valA, valB, valB/3060)

        try:
            row = [time.ctime(), time.time(), valA, valB]            
            with open('log-raw.csv', 'a') as f:
                w = csv.writer(f)
                w.writerow(row)
        except Exception as e:
            print(e)
            
        time.sleep(1)
except Exception as e:
    print(e)
# plot data from channel A and B
markers_on = [idx]
plt.plot(timebase, adc2mVChA[:],'-gD', markevery=markers_on)
plt.plot(timebase, adc2mVChB[:],'-bD', markevery=markers_on)
plt.plot(timebase, fit[:],'-rD', markevery=markers_on)
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (mV)')
plt.show()

# Stop the scope
# handle = chandle
status["stop"] = ps.ps2000_stop(chandle)
assert_pico2000_ok(status["stop"])

# Close unitDisconnect the scope
# handle = chandle
status["close"] = ps.ps2000_close_unit(chandle)
assert_pico2000_ok(status["close"])

# display status returns
print(status)
