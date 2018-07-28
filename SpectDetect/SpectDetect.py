#!/usr/bin/python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Module to detect earthquakes based on the energy arriving within a certain frequency band at multiple stations.
# Also allows for filtering events based on whether they exhibit dispersive arrivals (if dispersive, will not trigger a detection).


# Input variables:
    # Compulsary arguments:
    # jul_day - julian day to search over
    # hour - hour to search over
    # datadir - Parent directory containing mseed data (e.g. ./VITAL_DATA/MSEED/500sps/2014)
    # outdir - Directory to save outputs to
    # stations_to_use - List of stations to use (e.g. ["SKR01","SKR02","SKR03","SKR04","SKR05"])
    # fract_stations_filter - Minimum number of stations that have to detect an arrival before triggering an event detection
    # Optional arguments:
    # freq_range - Range of frequencies over which to calculate the average energy used as the detection parameter (e.g. [45,200])
    # min_snr - The minimum signal to noise ratio of the energy arriving within freq. band freq_range for triggering a detection
    # centre_freq_range - Range of centre frequencies over which to do FTAN dispersion filtering (Hz) (e.g. [4.0,200.0])
    # centre_freq_range_step - Size of centre frequency step used in dispersion filter (Hz) (e.g. 0.5)
    # band_width_gau_filter - Band width of the gaussian filter applied during the dispersion filtering (Hz) (e.g. [1.25])
    # plotSwitch - Boolian defining whether or not to produce key plots to check processing steps
    # dispersionFilterSwitch - Boolian defining whether to implement disperison filtering or not
    # returnCutMSEEDSwitch - Boolian defining whether or not to return cut mseed data for each event

# Output variables:

# Created by Tom Hudson, 29th Nov 2016

# NOTE:
# Run in colorado env (source activate colorado)
# Currently doesn't use instrument gains
# mseed must have components specified. (i.e. must have Z,N,E specified in trace stats)
# Currently only does 1 second detection windows.

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import obspy
import sys, os
from obspy.core import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import gps2DistAzimuth # Function to calc distance between two coordinates (see help(gps2DistAzimuth) for more info)
#from obspy.geodetics import gps2dist_azimuth as gps2DistAzimuth
from scipy import fft
import scipy
import matplotlib.mlab as mlab
from scipy.signal import hilbert # For FTAN analysis
from scipy.optimize import curve_fit
#import pandas # For rolling window in normallised cross-correlation
import copy

# # Define any constants:
# matplotlib_vs_obspy_plotting_switch = 2 # 1 for matplotlib plotting, 2 for obspy plotting
# #filename = '2014180_1800_total_mseed.m'
# #statName='SKR01'
# #CMM_events_list_input_for_rotation_filename = indir+'/'+'CMM_events_list_input_for_rotation.txt'
#  #["SKR01","SKR02","SKR03","SKR04","SKR05","SKR06","SKR07"] # List of stations to plot
# instGainFilename = "instrument_gain_data.txt" # File containing instrument gain data, in format: station, sensor gain (Z,N,E), digitaliser gain (Z,N,E)

# -------------------------------Define Main Functions In Script-------------------------------

def get_single_mseed_file_data(mseed_fname):
    '''Function to get mseed data given a mseed file or a set of files defined by wildcards.
    Returns st - stream containing detrended mseed data.''' 
    st = obspy.read(mseed_fname)
    st.detrend(type='demean') # The mean of the data is subtracted from the stream
    return st


def correctForInstrumentGain(st, instGainFilename, stations_array_to_plot):
    '''Function to correct for instrument gain, converting data from counts to velocity. Inputs are the uncorrected stream and a file containing stations and their gains.'''
    
    # Import instrument gain values:
    inst_gain_array = np.loadtxt(instGainFilename,dtype=str)
    
    # Create new empty stream to add data to:
    st_corrected = obspy.read()
    st_corrected.clear()
    
    # Loop over stations, correcting for sensor and digitalisor gains:
    for station in stations_array_to_plot:
        # Get gains:
        row_index_for_stat = np.where(inst_gain_array==station)[0][0]
        sensor_gain_Z_tmp = np.float(inst_gain_array[row_index_for_stat,1])
        sensor_gain_N_tmp = np.float(inst_gain_array[row_index_for_stat,2])
        sensor_gain_E_tmp = np.float(inst_gain_array[row_index_for_stat,3])
        digitaliser_gain_Z_tmp = np.float(inst_gain_array[row_index_for_stat,4])
        digitaliser_gain_N_tmp = np.float(inst_gain_array[row_index_for_stat,5])
        digitaliser_gain_E_tmp = np.float(inst_gain_array[row_index_for_stat,6])
        # Get traces (if possible):
        try:
            tr_tmp_Z = st.select(station=station).select(component="Z")[0]
            tr_tmp_N = st.select(station=station).select(component="N")[0]
            tr_tmp_E = st.select(station=station).select(component="E")[0]
            # Write traces to stream, corrected for:
            tr_tmp_Z.data = tr_tmp_Z.data/(sensor_gain_Z_tmp*digitaliser_gain_Z_tmp)
            tr_tmp_N.data = tr_tmp_N.data/(sensor_gain_N_tmp*digitaliser_gain_N_tmp)
            tr_tmp_E.data = tr_tmp_E.data/(sensor_gain_E_tmp*digitaliser_gain_E_tmp)
            st_corrected.append(tr_tmp_Z)
            st_corrected.append(tr_tmp_N)
            st_corrected.append(tr_tmp_E)
        except:
            continue
                
    return st_corrected
    

def whitened_spectrogram(st, statName, matplotlib_vs_obspy_plotting_switch, verbosity_level=0):
    """Function to get and plot whitened spectrogram for a particular station, and compare to unwhitened spectrogram."""
    
    data_processing_switch = False
    try:
        tr_Z_spec_stat = st.select(station=statName).select(component="Z")[0]
        data_processing_switch = True
    except:
        if verbosity_level==2:
            print "cannot get data for station: "+statName+" therefore continuing."
    
    if data_processing_switch:
        # Get amplitude spectral density (ASD) (unwhitened) (normalised to 1):
        # Process data:
        data = tr_Z_spec_stat.data
        fs = tr_Z_spec_stat.stats.sampling_rate
        NFFT = int(fs)*1 # Number of values used in each block (use int(fs)*0.05 for 90 sec data and int(fs)*1 for hour long data)
        Pxx , freqs = mlab.psd(data, Fs=fs, NFFT=NFFT) #calculate the amplitude spectral density
        # plt.figure()
        # plt.plot(freqs, np.sqrt(Pxx)/np.max(np.sqrt(Pxx)),color='b',label="ASD, before whitening")

        # Perform spectral whitening:
        psd = scipy.interpolate.interp1d(freqs, Pxx) # interpolates the ASD (unwhitened)
        freqs = np.fft.rfftfreq(len(data), 1/fs) # get the discrete fourier transform sample frequencies
        f_data = np.fft.rfft(data) #transform data to freqency domain
        white_f_data = f_data / (np.sqrt(psd(freqs) * fs /2.)) #divide by ASD (in frequency domain)
        white_data = np.fft.irfft(white_f_data, n=len(data)) #transform back into time domain with inverse fft

        # And plot unwhitened vs. whitened data ASD (to check that the data is indeed whitened!)
        Pxx_white_data , freqs_white_data = mlab.psd(white_data, Fs=fs, NFFT=NFFT) #calculate the amplitude spectral density, whitened data
        # plt.plot(freqs_white_data, np.sqrt(Pxx_white_data)/np.max(np.sqrt(Pxx_white_data)),color='r',label="ASD, after whitening")
        # plt.yscale('log')
        # plt.grid('on')
        # plt.ylabel('ASD, Normalised')
        # plt.xlabel('Freq (Hz)')
        # plt.legend()
        # plt.show()

        # And plot spectrogram of data:
        spec_cmap='viridis' #'viridis' #colormap
        
        # Plot using matplotlib:
        if matplotlib_vs_obspy_plotting_switch == 1:
            #plot a spectrogram
            x_window = [0, 3600] #define a window of the spectrogram to show
            NFFT=int(NFFT*1)
            noverlap = int(NFFT/2)
            #fig, axes = plt.subplots((2, figsize=(12, 8), sharex=True)
            plt.figure(figsize=(24, 12))
            ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
            ax2 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
            ax3 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
            #ax4 = plt.subplot2grid((2, 3), (1, 2), colspan=1)
            # For unwitened data:
            plt.sca(ax1) # Set currrent axis
            #np.hamming(NFFT) # Specify Hamming window using NFFT
            spec, freqs, bins, im = plt.specgram(data, NFFT=NFFT, noverlap=noverlap, Fs=fs, mode='psd', scale='dB', cmap=spec_cmap, xextent=[0,3600])
            # plt.colorbar(label='Decibel')
            plt.colorbar(label='Arb. units')
            plt.title(statName+" Unwhitened data, linear plot")
            plt.ylabel('Frequency (Hz)')
            plt.axis([x_window[0], x_window[1], 0, 250]) # The data is likely noisy below 10 Hz and above 100 Hz
            # For whitened data:
            plt.sca(ax2) # Set currrent axis
            spec, freqs, bins, im = plt.specgram(white_data, NFFT=NFFT, noverlap=noverlap, Fs=fs, mode='psd', scale='dB', cmap=spec_cmap, xextent=[0,3600]) #scale='dB', wlen=0.02
            #plt.colorbar(label='Decibel')
            plt.colorbar(label='Arb. units')
            plt.title(statName+" Unwhitened data, linear plot")
            plt.xlabel('Seconds since '+str(tr_Z_spec_stat.stats.starttime))
            plt.ylabel('Frequency (Hz)')
            plt.axis([x_window[0], x_window[1], 0, 250]) # The data is likely noisy below 10 Hz and above 100 Hz

            # # And plot ASD for data:
            # plt.sca(ax3)
            # data_event_only = data[int(500*10):500*11]
            # white_data_event_only = white_data[500*9:500*11]
            # Pxx_data_event_only , freqs_data_event_only = mlab.psd(data_event_only, Fs=fs, NFFT=NFFT)
            # Pxx_white_data_event_only , freqs_white_data_event_only = mlab.psd(white_data_event_only, Fs=fs, NFFT=NFFT)
            # plt.plot(freqs_data_event_only, np.sqrt(Pxx_data_event_only)/np.max(np.sqrt(Pxx_data_event_only)),color='r',label="ASD, unwhitened")
            # plt.plot(freqs_white_data_event_only, np.sqrt(Pxx_white_data_event_only)/np.max(np.sqrt(Pxx_white_data_event_only)),color='b',label="ASD, whitened")
            # plt.yscale('log')
            # plt.grid('on')
            # plt.ylabel('ASD, Normalised')
            # plt.xlabel('Freq (Hz)')
            # plt.title("ASD for 10-11 secs")
            # plt.legend()
            
        if verbosity_level==2:
            print "saving data for "+statName
        plt.savefig(statName+"_Z_spectrogram_comparison_hour_data.png")
        #plt.show()
        #plt.clf()
        
    return white_data


def icequake_search_using_spectrogram(st, statName_list, freq_range, min_snr, verbosity_level=0):
    """Function to search for icequakes using spectrogram. Won't pick up all events but will pick up a number for cross-correlation. Input is a stream, and a list of station names that are in the stream to be processed.
    Returns detection_all_stations_bins, an array of length number of seconds of data input (3600). This array contains th fraction of total stations that detected an event in the spectrogram for a given second. This can then be used to search for an exact event origin time."""
        
    # Create arrays for storing data:
    num_seconds_of_data = int(len(st[0].data)/st[0].stats.sampling_rate) #3600
    detection_all_stations_bins = np.zeros(num_seconds_of_data,dtype=float) # array to store detections at stations. A value of 1 is appended to a specific bin if an event is detected there, above the snr threshold specified below
    
    # Loop over all stations, getting spectrum and appending to total spectrum array:
    for i in range(len(statName_list)):
        statName = statName_list[i]
        if verbosity_level==2:
            print "Processing data for station: ", statName
        
        # Get spectrum data for station (if station exists):
        data_processing_switch = False
        try:
            tr_Z_spec_stat = st.select(station=statName).select(component="Z")[0]
            data_processing_switch = True
        except:
            if verbosity_level==2:
                print "cannot get data for station: "+statName+" therefore continuing."
        if data_processing_switch:
            # Get amplitude spectral density (ASD) (unwhitened) (normalised to 1):
            # Process data:
            data = tr_Z_spec_stat.data
            fs = tr_Z_spec_stat.stats.sampling_rate
            NFFT = int(fs)*1 # Number of values used in each block (use int(fs)*0.05 for 90 sec data and int(fs)*1 for hour long data)
            Pxx , freqs = mlab.psd(data, Fs=fs, NFFT=NFFT) #calculate the amplitude spectral density

            # Perform spectral whitening:
            psd = scipy.interpolate.interp1d(freqs, Pxx) # interpolates the ASD (unwhitened)
            freqs = np.fft.rfftfreq(len(data), 1/fs) # get the discrete fourier transform sample frequencies
            f_data = np.fft.rfft(data) #transform data to freqency domain
            white_f_data = f_data / (np.sqrt(psd(freqs) * fs /2.)) #divide by ASD (in frequency domain)
            white_data = np.fft.irfft(white_f_data, n=len(data)) #transform back into time domain with inverse fft
            
            # Get spectrogram data (would be better to use plt.specgram, as in tutorial, but doesnt work on uranus!):
            Pxx, freqs, t = mlab.specgram(white_data, NFFT=NFFT, Fs=fs) # Variables not used, that are in pyplot specgram fctn: mode='psd', scale='dB', cmap=spec_cmap, xextent=[0,3600], vmin=-30, vmax=-5
            spec = Pxx
            bins = t # bins equals t - t is array of mid point values of time in spectrogram
            
            # Find peaks in spectrum:
            #freq_range = [50,120] # Frequency range to search over
            freq_min_idx = np.argmin(abs(freqs - freq_range[0]))  #get the frequency indices closest to our frequency choice
            freq_max_idx = np.argmin(abs(freqs - freq_range[1]))
            freq_av_power = np.zeros(len(bins))
            freq_max_power = np.zeros(len(bins))
            for bin_idx in range(len(bins)): #loop over all bins of the spectrogram (the x-axis)
                freq_av_power[bin_idx] = np.average(spec[freq_min_idx : freq_max_idx, bin_idx]) #calculate the mean power within the freqency range
            # And get peaks from freq_av_power array:
            peak_idx = scipy.signal.find_peaks_cwt(freq_av_power, np.array([1]), min_snr=min_snr) #find peaks within a window of size 1 with SNR>2.3 # (widths=[1] doesn't work due to scipy version on uranus!, so pass it a numpy array)
            #print('found events at bin #: ', peak_idx)
            #print('corresponds to second:', np.round(bins[peak_idx],1))
            if len(peak_idx)>0:
                peak_idx_in_seconds =  np.round(bins[peak_idx],1) # data location, to nearest second #(np.array(peak_idx)/len(bins))*peak_idx
            
                # Add detections to detection store for current station:
                detection_all_stations_bins[peak_idx_in_seconds.astype(int)] += 1
            
            # Plot detections, for all:
            # plt.figure(figsize=(13,3))
            # plt.plot(bins, freq_av_power)
            # plt.xlabel('seconds since '+str(tr_Z_spec_stat.stats.starttime))
            # plt.ylabel('normalized energy density')
            # plt.xlim([0,max(bins)])
            # plt.show()
            
    # plt.figure()
    # plt.plot(np.arange(len(detection_all_stations_bins)), detection_all_stations_bins)
    # plt.xlabel('Seconds since '+str(tr_Z_spec_stat.stats.starttime))
    # plt.ylabel('Number of stations that detect an event at a given time')
    # plt.show()
        
    # And return detection data for all stations:
    return detection_all_stations_bins


def adjust_for_event_second_overlap(detection_all_stations_bins):
    """Function to return proportion of stations detections in each second, with any detections within the following second moved to the previous second. Therefore deals with overlaps. Note: Only deals with detections of an overlap of maximum 1 second."""
    
    # Loop over detections array, shifting detections that fall over multiple seconds to the same second:
    for i in np.arange(1,len(detection_all_stations_bins)):
        # Adjust data, if two adjacent seconds contain detections:
        if (detection_all_stations_bins[i-1] >= 1) and (detection_all_stations_bins[i] >= 1):
            detection_all_stations_bins[i-1] = detection_all_stations_bins[i-1] + detection_all_stations_bins[i] # Move all detections to first second (of adjacent two seconds)
            detection_all_stations_bins[i] = 0 # And remove previous detection, as moved to previous second
            
    return detection_all_stations_bins


def icequake_search_using_spectrogram_with_freq_band_ratios_sub_second(st, statName_list, detection_all_stations_bins, fract_stations_filter, verbosity_level=0):
    """Function to detect events based upon frequency band average amplitude ratios (e.g. average freq in 50-120 Hz / average freq in 25-50 Hz)"""
    
    # Specify spectrum picking parameters for detailed spectrum first peak pick (may be different to in function icequake_search_using_spectrogram() ):
    freq_range_P = [50,120] #[50,120]
    freq_range_lower_range = [10,25]
    min_snr=2.25 #2.0 #2.5 #2.3
    
    st_to_process=st.copy()
    
    # Get start time of trace:
    startTime = st[0].stats.starttime
    
    # Process spectogram data, for N component:
    # Loop through stations:
    for i in range(len(statName_list)):
        statName = statName_list[i]
        if verbosity_level==2:
            print "Testing for amplitude ratios for station: ", statName
        # Get spectrum data for station (if station exists):
        data_processing_switch = False
        try:
            tr_Z_spec_stat = st_to_process.select(station=statName).select(component="Z")[0]
            data_processing_switch = True
        except:
            if verbosity_level==2:
                print "cannot get data for station: "+statName+" therefore continuing."
        if data_processing_switch:
            
            # --------------------------- Process data for Z component (for P arrival) ---------------------------
            freq_range = freq_range_P
            
            # Whiten data (for entire hour):
            # Get amplitude spectral density (ASD) (unwhitened) (normalised to 1):
            # Process data:
            data = tr_Z_spec_stat # tr_Z_spec_stat.data+tr_N_spec_stat.data+tr_E_spec_stat.data
            fs = tr_Z_spec_stat.stats.sampling_rate
            NFFT = int(fs)*1 # Number of values used in each block (very low for looking at a few seconds of data)
            Pxx , freqs = mlab.psd(data, Fs=fs, NFFT=NFFT) #calculate the amplitude spectral density

            # Perform spectral whitening:
            psd = scipy.interpolate.interp1d(freqs, Pxx) # interpolates the ASD (unwhitened)
            freqs = np.fft.rfftfreq(len(data), 1/fs) # get the discrete fourier transform sample frequencies
            f_data = np.fft.rfft(data) #transform data to freqency domain
            white_f_data = f_data / (np.sqrt(psd(freqs) * fs /2.)) #divide by ASD (in frequency domain)
            white_data = np.fft.irfft(white_f_data, n=len(data)) #transform back into time domain with inverse fft
            

            # Loop over detections array, doing detailed spectrogram around event if enough stations record it:
            for a in range(len(detection_all_stations_bins)):
                # Only process if detected on sufficient number of stations:
                if detection_all_stations_bins[a]>=fract_stations_filter:
                    # Only process if have enough seconds before and after event:
                    if a>=1 and a<3599:
                        if verbosity_level==2:
                            print "Testing for amplitude ratios at time:", startTime+a
                        # Trim data for specific current event:
                        start_index = int(fs*(a-1))
                        end_index = int(fs*(a+1))
                        white_data_spec_event_trimmed = white_data[start_index:end_index]
                        #st_tmp_trimmed = st_to_process.trim(starttime=startTime+a-1.0, endtime=startTime+a+2.0)
                        
                        # Get spectrogram data (would be better to use plt.specgram, as in tutorial, but doesnt work on uranus!):
                        # ORIGINAL: Pxx, freqs, t = mlab.specgram(white_data, NFFT=10, Fs=fs,noverlap=5) # Variables not used, that are in pyplot specgram fctn: mode='psd', scale='dB', cmap=spec_cmap, xextent=[0,3600], vmin=-30, vmax=-5
                        NFFT = int(fs)*1/25 #*1/20 #1/25 # Number of values used in each block (very low for looking at a few seconds of data)
                        per_lap=0.9 # Percentage overlap value
                        nlap = int(NFFT * float(per_lap))
                        spec, freqs, time = mlab.specgram(white_data_spec_event_trimmed, Fs=fs, NFFT=NFFT, noverlap=nlap)
                        bins = time # bins equals t - t is array of mid point values of time in spectrogram
            
                        # Find first peak in detailed 3s spectrum:
                        # Get  ratio over frequency range average that might contain P arrival/lower freq range:
                        freq_min_idx = np.argmin(abs(freqs - freq_range_P[0]))  #get the frequency indices closest to our frequency choice
                        freq_max_idx = np.argmin(abs(freqs - freq_range_P[1]))
                        freq_min_idx_lower_range = np.argmin(abs(freqs - freq_range_lower_range[0]))
                        freq_max_idx_lower_range = np.argmin(abs(freqs - freq_range_lower_range[1]))
                        freq_av_power_ratio = np.zeros(len(bins))
                        freq_max_power = np.zeros(len(bins))
                        for bin_idx in range(len(bins)): #loop over all bins of the spectrogram (the x-axis)
                            freq_av_power_ratio[bin_idx] = np.average(spec[freq_min_idx : freq_max_idx, bin_idx])/np.average(spec[freq_min_idx_lower_range : freq_max_idx_lower_range, bin_idx]) #calculate the mean power ratio
                        
                        # Plot spectrum:
                        fig, (ax0, ax1) = plt.subplots(2, sharex=True, figsize=(8,6))
                        # Plot waveform:
                        tr_Z_spec_stat_filt = tr_Z_spec_stat.copy()
                        tr_Z_spec_stat_filt.filter('bandpass', freqmin=freq_range_P[0], freqmax=freq_range_P[1], corners=4, zerophase=True)
                        y_tmp = tr_Z_spec_stat_filt.data[start_index:end_index] # Cut out only waveform data for 2 seconds
                        x_tmp = np.linspace(-1000,1000,num=len(y_tmp))
                        ax0.plot(x_tmp,y_tmp, color='k')
                        ax0.set_ylabel("Amplitude, N_comp, filtered (counts)")
                        # And plot spectrogram:
                        #ax1.imshow(spec[:,int(len(time)/3):int(2*len(time)/3)], cmap="hot", origin='lower', extent=(0, 1000, freqs[0], freqs[-1]))
                        ax1.imshow(spec[:,:], cmap="hot", origin='lower', extent=(-1000, 1000, freqs[0], freqs[-1]), aspect="auto")
                        ax1.set_xlabel("Time after "+str(startTime+a)+" (ms)")
                        ax1.set_ylabel("Frequency (Hz)")
                        # Plot frequency band, for data used for detection:
                        ax1.axhline(y=freq_range_P[0],color='cyan', linestyle="dashed")
                        ax1.axhline(y=freq_range_P[1],color='cyan', linestyle="dashed")
                        ax1.axhline(y=freq_range_lower_range[0],color='blue', linestyle="dashed")
                        ax1.axhline(y=freq_range_lower_range[1],color='blue', linestyle="dashed")
                        # And plot ratio data used for detection:
                        ax2 = ax1.twinx() # Set up second y scale
                        y_tmp = freq_av_power_ratio[:] # Get 1 second of average frequency power data
                        x_tmp = np.linspace(-1000,1000,num=len(y_tmp)) # Get x values to correspond with ms scale of spectrogram plot
                        ax2.plot(x_tmp, y_tmp, color='g')
                        ax2.set_ylabel("Average power ratio (arb. units)")
                        # plt.plot(spec[:,400], freqs)
                        fig.suptitle("P arrival")
                        plt.show()


def basic_FTAN_calc_and_plot(real_waveform, centre_freq_range, centre_freq_range_step, band_width_gau_filter, sampling_rate, plotSwitch=True):
    """perform basic FTAN analysis on a signal. Will output a frequency/time plot rather than a group velocity/period plot (which is the default display for FTAN analysis).
    Inputs are: real_waveform - real waveform assocaited with potential surface wave; centre_freq_range - range of centre frequencies to loop over; centre_freq_range_step - Step size for centre frequencies to be looped over; band_width_gau_filter - bandwidth of basic gaussian filter.
    Outputs are: S - FTAN array, in frequency and waveform time; plot of S - The FTAN array, with respect to frequency and time."""
    
    # 1. Get Hilbert transform of real_waveform:
    W_t_domain = hilbert(real_waveform) #scipy.signal.hilbert(real_waveform) # Outputs analytical solution with real and imaginary parts (x_a = x + yi) (- can get real and imaginary parts using np.real, np.imag)
    
    # 2. Take FFT of W(t) to find K(w):
    K_f_domain = np.fft.fft(W_t_domain)/(len(W_t_domain)**0.5) # Normalised by root(n)
    
    # Get frequency array associated with signal:
    freqs = np.fft.fftfreq(len(W_t_domain), d=(1/sampling_rate))
    freqs_rad_per_s = freqs*2*np.pi # Frequency in radians per second
    
    # Get FTAN output:
    centre_freqs_array = np.arange(centre_freq_range[0], centre_freq_range[1], centre_freq_range_step)
    time_array = np.linspace(0,len(real_waveform)/sampling_rate,num=len(real_waveform))
    # Specify array to store output FTAN data:
    S_t_domain_array = np.zeros((len(real_waveform), len(centre_freqs_array))).astype(complex) # Array containing [time along trace x centre frequencies]
    # Loop over central frequencies:
    for i in np.arange(len(centre_freqs_array)):
        w_H = centre_freqs_array[i]*2*np.pi # Current centre frequency to work on
        # Get bandwidth for current central frequency:
        try:
            band_width_w_H_tmp = band_width_gau_filter[i]*2*np.pi # bandwidth in rad
        except:
            band_width_w_H_tmp = band_width_gau_filter[0]*2*np.pi  # bandwidth in rad
        # 3. Get gaussian filter function in frequency domain, for specific centre frequency:
        Gauss_filt_freq_domain = np.zeros(len(freqs_rad_per_s)).astype(complex)
        for j in np.arange(len(freqs_rad_per_s)):
            Gauss_filt_freq_domain[j] = (1/(((2*np.pi)**0.5)*band_width_w_H_tmp))*np.exp(-1*((freqs_rad_per_s[j] - w_H)**2)/(2*(band_width_w_H_tmp**2)))
        # 4. Then find FTAN values for current centre freq:
        S_f_domain = Gauss_filt_freq_domain*K_f_domain
        S_t_domain_array[:, i] = np.fft.ifft(S_f_domain)*(len(W_t_domain)**0.5) # Normalised by root(n)
                
    if plotSwitch:
        # Plot results:
        plt.figure(figsize=[8,6])
        #y, x = np.mgrid[np.linspace(0,len(real_waveform)/sampling_rate,num=len(real_waveform)), freqs]
        plt.imshow(np.transpose(np.absolute(S_t_domain_array)), cmap="jet", origin='lower')#, extent=(0.0, len(real_waveform)/sampling_rate, centre_freqs_array[0], centre_freqs_array[-1]), aspect="auto")
        plt.show()
    
        # And plot comparison to spectrogram and waveform:
        fs = sampling_rate
        NFFT = int(fs)*1/5 #*1/20 #1/25 # Number of values used in each block (very low for looking at a few seconds of data)
        per_lap=0.9 # Percentage overlap value
        nlap = int(NFFT * float(per_lap))
        spec, freqs, time = mlab.specgram(real_waveform, Fs=fs, NFFT=NFFT, noverlap=nlap)
        fig, (ax0, ax1, ax2) = plt.subplots(3, sharex=True, figsize=(12,8))
        x_tmp = np.linspace(0,len(real_waveform)/sampling_rate,num=len(real_waveform))
        ax0.plot(x_tmp, real_waveform, color='k')
        ax1.imshow(np.transpose(np.absolute(S_t_domain_array)), cmap="jet", origin='lower', extent=(0.0, len(real_waveform)/sampling_rate, centre_freqs_array[0], centre_freqs_array[-1]), aspect="auto")
        #ax1.imshow(np.transpose(np.real(S_t_domain_array)), cmap="jet", origin='lower', extent=(0.0, len(real_waveform)/sampling_rate, centre_freqs_array[0], centre_freqs_array[-1]), aspect="auto")
        ax2.imshow(spec[:,:], cmap="hot", origin='lower', extent=(0.0, len(real_waveform)/sampling_rate, freqs[0], freqs[-1]), aspect="auto")
        ax2.set_xlabel("Time (s)")
        ax0.set_ylabel("Amplitude (counts)")
        ax1.set_ylabel("Centre frequency (Hz)")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_ylim([0,100])
        plt.show()
    
    return S_t_domain_array, centre_freqs_array, time_array


def get_dispersion_curve_from_FTAN_array(S_t_domain_array, centre_freqs_array, time_array, plotSwitch=True, verbosity_level=0):
    """Function to get dispersion curve from FTAN array."""
    
    # Find maximum centre freq for all time:
    max_centre_freq_with_time_array = np.zeros(len(time_array),dtype=float)
    max_centre_freq_with_time_array_GAU = np.zeros(len(time_array),dtype=float)
    # Loop through FTAN array to find maximum centre freq at each time:
    for i in range(len(time_array)):
        # Just taking maximum:
        max_centre_freq_index_tmp = np.argmax(np.absolute(S_t_domain_array[i, :])) # Index of max centre freq (of absolute magnitude)
        max_centre_freq_with_time_array[i] = centre_freqs_array[max_centre_freq_index_tmp] # Get max centre freq from labelling array
        # Taking Gaussian fit then maximum:
        def gau_func(x,b,c):
            return (np.exp(-1*((x-b)**2)/(2*(c**2))))
        xdata = centre_freqs_array
        ydata = np.absolute(S_t_domain_array[i, :])
        popt, pcov = curve_fit(gau_func, xdata, ydata) # Fit the gaussian to the data
        res_factor = 25 # Factor to up resolution by
        xdata_high_res = np.linspace(centre_freqs_array[0], centre_freqs_array[-1], len(centre_freqs_array)*res_factor)
        max_centre_freq_high_res_index_tmp_gau = np.argmax(gau_func(xdata_high_res, *popt))
        # Write corresponding centre freq to data storage array (if meets correct filters):
        if xdata_high_res[max_centre_freq_high_res_index_tmp_gau]>np.min(centre_freqs_array) and xdata_high_res[max_centre_freq_high_res_index_tmp_gau]<(np.max(centre_freqs_array)/2):
            max_centre_freq_with_time_array_GAU[i] = xdata_high_res[max_centre_freq_high_res_index_tmp_gau] # Get max centre freq from labelling array
        else:
            max_centre_freq_with_time_array_GAU[i] = np.nan # As outside realistic range so dont want to plot or use
    
    # And calculate gradient from guassian fitting method:
    max_centre_freq_with_time_array_GAU_NAN_values_removed = max_centre_freq_with_time_array_GAU[np.isfinite(max_centre_freq_with_time_array_GAU)] # Remove nan values
    max_centre_freq_with_time_array_GAU_NAN_values_removed_gradient = np.gradient(max_centre_freq_with_time_array_GAU_NAN_values_removed)
    av_gradient = np.average(max_centre_freq_with_time_array_GAU_NAN_values_removed_gradient)
    if verbosity_level==2:
        print "average gradient:", av_gradient
        print "Positive gradient, dispersed. Negative gradient, not dispersive."
    
    # Calculate length of positive and negative parts of fit:
    len_pos_grad_of_fit = len(max_centre_freq_with_time_array_GAU_NAN_values_removed_gradient[max_centre_freq_with_time_array_GAU_NAN_values_removed_gradient>=0.0])
    len_neg_grad_of_fit = len(max_centre_freq_with_time_array_GAU_NAN_values_removed_gradient[max_centre_freq_with_time_array_GAU_NAN_values_removed_gradient<0.0])
    if verbosity_level==2:
        print "Length +ve:", len_pos_grad_of_fit, "Length -ve:", len_neg_grad_of_fit
    
    # Calculates length of positive and negative parts of fit, ONLY from times with normalised amplitude greater than 0.5:
    if verbosity_level==2:
        print "Finding dispersion gradient ONLY for norm. amp. > 0.7:"
    # Normalise FTAN array:
    S_t_domain_array_normalised = np.absolute(S_t_domain_array)/np.max(np.absolute(S_t_domain_array))
    max_centre_freq_with_time_array_GAU_NAN_values_removed_filtered_by_amplitude = np.empty(0, dtype=float) # empty array to fill data with
    # Find indices in time where normalised FTAN array > 0.5, and write gaussian fit value to array if true:
    S_t_domain_array_max_amplitude_array = np.zeros(len(time_array), dtype=float)
    for j in range(len(time_array)):
        tmp_S_t_domain_array_max_amplitude = np.max(S_t_domain_array_normalised[j, :]) # Fill array
        S_t_domain_array_max_amplitude_array[j] = tmp_S_t_domain_array_max_amplitude
        # Get gaussian fit for only these regions if amplitude > 0.5:
        if tmp_S_t_domain_array_max_amplitude > 0.7:
            max_centre_freq_with_time_array_GAU_NAN_values_removed_filtered_by_amplitude = np.append(max_centre_freq_with_time_array_GAU_NAN_values_removed_filtered_by_amplitude, max_centre_freq_with_time_array_GAU[j])
    # And remove any values previously filtered for being unrealistic:
    max_centre_freq_with_time_array_GAU_NAN_values_removed_filtered_by_amplitude = max_centre_freq_with_time_array_GAU_NAN_values_removed_filtered_by_amplitude[np.isfinite(max_centre_freq_with_time_array_GAU_NAN_values_removed_filtered_by_amplitude)]
    # And find gradient:
    max_centre_freq_with_time_array_GAU_NAN_values_removed_filtered_by_amplitude_gradient = np.gradient(max_centre_freq_with_time_array_GAU_NAN_values_removed_filtered_by_amplitude)
    # And print length results:
    # Calculate length of positive and negative parts of fit:
    len_pos_grad_of_fit_filtered_by_amplitude = len(max_centre_freq_with_time_array_GAU_NAN_values_removed_filtered_by_amplitude_gradient[max_centre_freq_with_time_array_GAU_NAN_values_removed_filtered_by_amplitude_gradient>=0.0])
    len_neg_grad_of_fit_filtered_by_amplitude = len(max_centre_freq_with_time_array_GAU_NAN_values_removed_filtered_by_amplitude_gradient[max_centre_freq_with_time_array_GAU_NAN_values_removed_filtered_by_amplitude_gradient<0.0])
    if verbosity_level==2:
        print "Length +ve:", len_pos_grad_of_fit_filtered_by_amplitude, "Length -ve:", len_neg_grad_of_fit_filtered_by_amplitude
    
    if plotSwitch:
        # And plot:
        fig, (ax1) = plt.subplots(1, sharex=True, figsize=(8,6))
        im = ax1.imshow(np.transpose(np.absolute(S_t_domain_array)), cmap="jet", origin='lower', extent=(time_array[0], time_array[-1], centre_freqs_array[0], centre_freqs_array[-1]), aspect="auto")
        plt.colorbar(im)
        ax1.plot(time_array, max_centre_freq_with_time_array, color='k',linewidth=2.5, alpha=0.5, label="Raw max")
        ax1.plot(time_array, max_centre_freq_with_time_array_GAU, color='g',linewidth=2.5, alpha=0.5, label="Gaussian fit max")
        x_tmp = time_array
        y_tmp = S_t_domain_array_max_amplitude_array
        ax1.fill_between(x_tmp, 0, 1, where=y_tmp > 0.7, facecolor='red', alpha=0.5)
        ax1.set_ylabel("Centre frequency (Hz)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylim((0,75))
        plt.legend()
        plt.show()
    
    return av_gradient, len_pos_grad_of_fit, len_neg_grad_of_fit, len_pos_grad_of_fit_filtered_by_amplitude, len_neg_grad_of_fit_filtered_by_amplitude


def FTAN_filter_for_dispersion(st, detection_all_stations_bins, centre_freq_range, centre_freq_range_step, band_width_gau_filter, sampling_rate, statName_list, fract_stations_filter, plotSwitch, verbosity_level=0):
    """Function to run FTAN analysis, obtaining a gradient for the S wave frequency content with time. If gradient is positive, then assume dispersive surface wave arrival detected and discard event.
    Runs for 1 hour data stream at a time, only looking at events that have triggereed on a sufficient number of stations.
    Filters out events by removing station contribution to detection_all_stations_bins array for given time.
    Full theoretical description on Mac in dir: /Users/tomhudson/Python/obspy_scripts/FTAN_basic_script_DEVELOPMENT."""    
    
    # Make copy of stream:
    st_to_process=st.copy()
    # Get start time of trace:
    startTime = st_to_process[0].stats.starttime
    
    # Process data, for N component, for each event:
    # Loop over detections array, doing detailed FTAN around event if enough stations record it:
    for a in range(len(detection_all_stations_bins)):
        # Only process if detected on sufficient number of stations:
        if detection_all_stations_bins[a]>=fract_stations_filter:
            # Only process if have enough seconds before and after event:
            if a>=1 and a<3599:
                if verbosity_level==1:
                    print "Testing for dispersion for event at time:", startTime+a
                # Loop through stations:
                for i in range(len(statName_list)):
                    statName = statName_list[i]
                    tr_spec_station_tmp = st_to_process.select(station=statName).select(component="N")[0] # Trace for specific station
                    tr_spec_station_tmp = tr_spec_station_tmp.copy()
                    # Trim data for specific current event:
                    tr_spec_station_tmp.trim(starttime=startTime+a-1, endtime=startTime+a+2) # +1 or +2 for endtime depends upon overlap time...
                    # And get real waveform np array data from trace:
                    real_waveform = tr_spec_station_tmp.data # Data associated with trace of real waveform observed
                                        
                    # Perform FTAN algorithm:
                    S_t_domain_array, centre_freqs_array, time_array = basic_FTAN_calc_and_plot(real_waveform, centre_freq_range, centre_freq_range_step, band_width_gau_filter, sampling_rate, False)
                    # Get dispersion gradient from FTAN analysis:
                    try:
                        if verbosity_level==2:
                            print statName
                        av_gradient, len_pos_grad_of_fit, len_neg_grad_of_fit, len_pos_grad_of_fit_filtered_by_amplitude, len_neg_grad_of_fit_filtered_by_amplitude = get_dispersion_curve_from_FTAN_array(S_t_domain_array, centre_freqs_array, time_array, plotSwitch)
                        
                        # Check if gradient is positive overall - if so, wave is dispersive, hence surface wave arrival, therefore remove from detection array:
                        #if av_gradient>0.0:
                        #if len_pos_grad_of_fit > len_neg_grad_of_fit:
                        if av_gradient>0.0 and len_pos_grad_of_fit_filtered_by_amplitude > len_neg_grad_of_fit_filtered_by_amplitude:
                            detection_all_stations_bins[a] = detection_all_stations_bins[a] - 1 # Remove station contribution
                                        
                    except:
                        if verbosity_level==2:
                            print "Could not obtain gradient for station: ",statName, " therefore continuing without removing."
                    
                    # # TEST ONLY!!!:
                    # print "TESTING BEGINNING"
                    # av_gradient, len_pos_grad_of_fit, len_neg_grad_of_fit = get_dispersion_curve_from_FTAN_array(S_t_domain_array, centre_freqs_array, time_array, plotSwitch)
                    # print "TESTING ENDING"
                    
                    del tr_spec_station_tmp # Clear previous history of the temparory variable, for next loop iteration
                     
    # And return updated detection array:
    return detection_all_stations_bins
    

def produce_CMM_event_output_files(st, detection_all_stations_bins, fract_stations_filter, outdir, verbosity_level=0):
    """Function to output approximate CMM line for event, based upon second detection accuracy only. Outputs CMM files"""
    
    # Get start time of hour data processed:
    hour_start_time = st[0].stats.starttime
    event_approx_seconds = np.where(detection_all_stations_bins>=fract_stations_filter)[0] # Get event seconds where above station fraction threshold
    event_approx_seconds = np.array(event_approx_seconds,dtype=float)
    
    if verbosity_level==1:
        print "Writing files to CMM output event line files"
        print "(For events found on more than", fract_stations_filter, "stations)"
    
    # Loop over detected events, writing to file:
    for event_time_in_sec in event_approx_seconds:
        event_time_tmp = st[0].stats.starttime + event_time_in_sec # Get start time for hour + event time in seconds.
        # Write event time, and artificial location, to file
        if verbosity_level==1:
            print "Writing:", str(event_time_tmp.year)+str(event_time_tmp.month).zfill(2)+str(event_time_tmp.day).zfill(2)+str(event_time_tmp.hour).zfill(2)+str(event_time_tmp.minute).zfill(2)+str(event_time_tmp.second).zfill(2)+".cmm", "to file"
        outfile = open(outdir+"/"+str(event_time_tmp.year)+str(event_time_tmp.month).zfill(2)+str(event_time_tmp.day).zfill(2)+str(event_time_tmp.hour).zfill(2)+str(event_time_tmp.minute).zfill(2)+str(event_time_tmp.second).zfill(2)+".cmm", 'w')
        outfile.write("#"+str(event_time_tmp.julday)+" "+str(event_time_tmp.year)+"-"+str(event_time_tmp.month).zfill(2)+"-"+str(event_time_tmp.day).zfill(2)+" "+str(event_time_tmp.hour).zfill(2)+":"+str(event_time_tmp.minute).zfill(2)+":"+str(event_time_tmp.second).zfill(2)+".0"+" 64.32799 -17.22406 -1.0 0 0 0 0 0 0")
        outfile.close()


def cut_mseed_for_events(st, detection_all_stations_bins, fract_stations_filter, outdir, verbosity_level=0):
    """Function to cut mseed for events, to 90 seconds, with event starting at 10 seconds. Saves data to outdir."""
    
    # Specify time before and after event time:
    time_dur_before = 10.0
    time_dur_after = 80.0
    
    # Get start time of hour data processed:
    hour_start_time = st[0].stats.starttime
    event_approx_seconds = np.where(detection_all_stations_bins>=fract_stations_filter)[0] # Get event seconds where above station fraction threshold
    event_approx_seconds = np.array(event_approx_seconds,dtype=float)
    
    if verbosity_level== 1:
        print "Writing files to cut mseed format, for events detected"
        print "(For events found on more than", fract_stations_filter, "stations)"
    
    # Loop over detected events, writing to file:
    for event_time_in_sec in event_approx_seconds:
        event_time_tmp = st[0].stats.starttime + event_time_in_sec # Get start time for hour + event time in seconds.
        # Write event time, and artificial location, to file
        if verbosity_level==2:
            print "Writing:", str(event_time_tmp.year)+str(event_time_tmp.month).zfill(2)+str(event_time_tmp.day).zfill(2)+str(event_time_tmp.hour).zfill(2)+str(event_time_tmp.minute).zfill(2)+str(event_time_tmp.second).zfill(2)+".m", "to file"
        filename_tmp = outdir+"/"+str(event_time_tmp.year)+str(event_time_tmp.month).zfill(2)+str(event_time_tmp.day).zfill(2)+str(event_time_tmp.hour).zfill(2)+str(event_time_tmp.minute).zfill(2)+str(event_time_tmp.second).zfill(2)+".m"
        # And prepping stream for writing to file:
        st_cut_tmp = st.copy()
        st_cut_tmp.trim(starttime=event_time_tmp-time_dur_before, endtime=event_time_tmp+time_dur_after)
        # Write cut stream to file:
        st_cut_tmp.write(filename_tmp, format="MSEED")
        # And remove stream:
        del st_cut_tmp
        
    if verbosity_level==1:
        print "Finished writing data to mseed."


def STA_LTA_detect_event_arrivals_from_spectrogram_station_detection_fract(st, statName_list, detection_all_stations_bins, fract_stations_filter, outdir, verbosity_level=0):
    """Function to detect event arrivals using STA/LTA arrival algorithm, using station detection fraction as a guide."""
    
    # Specify constants for doing STA/LTA analysis:
    STA_window = 0.01
    LTA_window = 0.5
    freqmin=20.0
    freqmax=100
    t_delay_min_P_to_S = 0.075 # Minimum time in seconds between P and S time
    
    # Filter detection_all_stations_bins array to dind events that are above fract_stations_filter:
    if verbosity_level==1:
        print "Filtering data for events found on more than", fract_stations_filter, "stations"
    event_approx_seconds = np.where(detection_all_stations_bins>=fract_stations_filter)[0] # Get event seconds where above station fraction threshold
    event_approx_seconds = np.array(event_approx_seconds,dtype=float)
    
    # Loop over each event detection, performing STA/LTa analysis for each event:
    for event_time_in_sec in event_approx_seconds:
        # Filter as cannot get data from another hour (this will mean we might miss events!):
        if event_time_in_sec>3 and event_time_in_sec<3597:
            #print st[0].stats.starttime, float(event_time_in_sec)
            event_time_tmp = st[0].stats.starttime + float(event_time_in_sec) # Get start time for hour + event time in seconds.
            
            # Create NLLoc obs file (to store picks in, for input into NLLoc then CMM locate):
            if verbosity_level==2:
                print "Found event and writing to file: ",str(event_time_tmp.year)+str(event_time_tmp.month).zfill(2)+str(event_time_tmp.day).zfill(2)+str(event_time_tmp.hour).zfill(2)+str(event_time_tmp.minute).zfill(2)+str(event_time_tmp.second).zfill(2)+".nonlinloc"
            outfile = open(outdir+"/"+str(event_time_tmp.year)+str(event_time_tmp.month).zfill(2)+str(event_time_tmp.day).zfill(2)+str(event_time_tmp.hour).zfill(2)+str(event_time_tmp.minute).zfill(2)+str(event_time_tmp.second).zfill(2)+".nonlinloc", 'w')
            
            # Loop through each station:
            for statName in statName_list:
                
                # ------------------- Pick P arrival -------------------
                st_tmp_Z = st.select(component="Z").select(station=statName)
                tr_tmp_Z = st_tmp_Z[0].trim(starttime=event_time_tmp-2.0,endtime=event_time_tmp+2.0)
                
                # STA/LTA picker (not standard, see below):
                tr=tr_tmp_Z
                # Filtered trace:
                tr_filt = tr.copy()
                tr_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
                # Find characteristic function (as in Allen et al 1978):
                characteristic_function_array = np.zeros(len(tr.data), dtype=float) # array to store characteristic_function_array (as in Allen et al 1978)
                weightingStationConst = 1.0 # C2 in Allen et al 1978
                for a in np.arange(len(tr.data)-1):
                    characteristic_function_array[a] = ((tr_filt[a])**2) + weightingStationConst + ((tr_filt[a+1]-tr_filt[a])**2)
    
                # Find STA/LTA of characteristic function (as in beamline STA/LTA method outlined at Seismix, final talk of Session 3, Tuesday):
                char_fnct_STA_LTA_array = np.zeros(len(tr.data), dtype=float) # array to store STA/LTA data
                # Find STA/LTA values:
                for i in np.arange(np.int(LTA_window*tr.stats.sampling_rate), len(tr.data)-np.int(tr.stats.sampling_rate*STA_window)-1):
                    # Note: i is the value of the index between the LTA and the STA.
                    # Find average STA and LTA window values for given point:
                    av_STA_window_temp = np.average(characteristic_function_array[i:(np.int(STA_window*tr.stats.sampling_rate) + i)]) # Find average STA value for a given point
                    j = i - np.int(LTA_window*tr.stats.sampling_rate) # To give first index where LTA starts (as i equals mid point between LTA and STA)
                    av_LTA_window_temp = np.average(characteristic_function_array[j:i]) # Find average LTA value for a given point
                    # Assign STA/LTA value to array:
                    char_fnct_STA_LTA_array[i] = av_STA_window_temp/av_LTA_window_temp
                
                # Pick based upon maximum of STA/LTA:
                try:
                    P_pick_secs_after_trace_start = np.argmax(char_fnct_STA_LTA_array)/tr.stats.sampling_rate
                    P_pick_time_tmp = event_time_tmp-2.0+P_pick_secs_after_trace_start
                    #print statName, "P", P_pick_time_tmp.date, P_pick_time_tmp.time, "SNR:",np.max(char_fnct_STA_LTA_array)
                    outfile.write(statName+"   ?    ?    ? P      ? "+str(P_pick_time_tmp.year)+str(P_pick_time_tmp.month).zfill(2)+str(P_pick_time_tmp.day).zfill(2)+" "+str(P_pick_time_tmp.hour).zfill(2)+str(P_pick_time_tmp.minute).zfill(2)+" "+str(P_pick_time_tmp.second).zfill(2)+"."+str(P_pick_time_tmp.microsecond)+" GAU 1.00e-02  0.00e+00  0.00e+00  0.00e+00  1.00e+00") # DYJS   ?    ?    ? P      D 20140823 0534 0.3500 GAU 1.00e-01  0.00e+00  0.00e+00  0.00e+00  1.00e+00 
                    outfile.write("\n")
                except:
                    continue
                
                # Plot data for parameter setting:
                # if statName == "SKR01":
                #     fig, axes = plt.subplots(3)
                #     axes[0].plot(range(len(tr_filt.data)),tr_filt.data)
                #     axes[0].plot(range(len(tr_filt.data)),tr_filt.data)
                #     axes[0].axvline(x=(P_pick_secs_after_trace_start*tr.stats.sampling_rate), ymin=0.25, ymax=0.75)
                #     axes[1].plot(range(len(char_fnct_STA_LTA_array)), char_fnct_STA_LTA_array)
                #     tr_filt.spectrogram(axes=axes[2],wlen=0.02)
                #     plt.ylabel('Amplitude')
                #     plt.xlabel('Time [s]')
                #     plt.show()
                
                # ------------------- Pick S arrival -------------------
                st_tmp_N = st.select(component="N").select(station=statName)
                tr_tmp_N = st_tmp_N[0].trim(starttime=event_time_tmp-2.0,endtime=event_time_tmp+2.0)
                
                # STA/LTA picker (not standard, see below):
                tr=tr_tmp_N
                # Filtered trace:
                tr_filt = tr.copy()
                tr_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
                # Find characteristic function (as in Allen et al 1978):
                characteristic_function_array = np.zeros(len(tr.data), dtype=float) # array to store characteristic_function_array (as in Allen et al 1978)
                weightingStationConst = 1.0 # C2 in Allen et al 1978
                for a in np.arange(len(tr.data)-1):
                    characteristic_function_array[a] = ((tr_filt[a])**2) + weightingStationConst + ((tr_filt[a+1]-tr_filt[a])**2)
    
                # Find STA/LTA of characteristic function (as in beamline STA/LTA method outlined at Seismix, final talk of Session 3, Tuesday):
                char_fnct_STA_LTA_array = np.zeros(len(tr.data), dtype=float) # array to store STA/LTA data
                # Find STA/LTA values:
                for i in np.arange(np.int(LTA_window*tr.stats.sampling_rate), len(tr.data)-np.int(tr.stats.sampling_rate*STA_window)-1):
                    # Note: i is the value of the index between the LTA and the STA.
                    # Find average STA and LTA window values for given point:
                    av_STA_window_temp = np.average(characteristic_function_array[i:(np.int(STA_window*tr.stats.sampling_rate) + i)]) # Find average STA value for a given point
                    j = i - np.int(LTA_window*tr.stats.sampling_rate) # To give first index where LTA starts (as i equals mid point between LTA and STA)
                    av_LTA_window_temp = np.average(characteristic_function_array[j:i]) # Find average LTA value for a given point
                    # Assign STA/LTA value to array:
                    char_fnct_STA_LTA_array[i] = av_STA_window_temp/av_LTA_window_temp
                
                # Pick based upon maximum of STA/LTA:
                # Remove STA/LTA values associated with P arrival energy:
                try:
                    char_fnct_STA_LTA_array[int((P_pick_secs_after_trace_start - t_delay_min_P_to_S)*tr.stats.sampling_rate):int((P_pick_secs_after_trace_start + t_delay_min_P_to_S)*tr.stats.sampling_rate)] = 0.0
                    S_pick_secs_after_trace_start = np.argmax(char_fnct_STA_LTA_array)/tr.stats.sampling_rate
                    # Check that S pick after P pick, otherwise remove:
                    while S_pick_secs_after_trace_start < P_pick_secs_after_trace_start:
                        char_fnct_STA_LTA_array[int((S_pick_secs_after_trace_start - t_delay_min_P_to_S)*tr.stats.sampling_rate):int((S_pick_secs_after_trace_start + t_delay_min_P_to_S)*tr.stats.sampling_rate)] = 0.0
                        S_pick_secs_after_trace_start = np.argmax(char_fnct_STA_LTA_array)/tr.stats.sampling_rate
                    # while True:
                    #     if S_pick_secs_after_trace_start < P_pick_secs_after_trace_start:
                    #         char_fnct_STA_LTA_array[int((S_pick_secs_after_trace_start - t_delay_min_P_to_S)*tr.stats.sampling_rate):int((S_pick_secs_after_trace_start + t_delay_min_P_to_S)*tr.stats.sampling_rate)] = 0.0
                    #         S_pick_secs_after_trace_start = np.argmax(char_fnct_STA_LTA_array)/tr.stats.sampling_rate
                    #     else:
                    #         break
                    S_pick_time_tmp = event_time_tmp-2.0+S_pick_secs_after_trace_start
                    #print statName, "S", S_pick_time_tmp.date, S_pick_time_tmp.time, "SNR:",np.max(char_fnct_STA_LTA_array)
                    outfile.write(statName+"   ?    ?    ? S      ? "+str(S_pick_time_tmp.year)+str(S_pick_time_tmp.month).zfill(2)+str(S_pick_time_tmp.day).zfill(2)+" "+str(S_pick_time_tmp.hour).zfill(2)+str(S_pick_time_tmp.minute).zfill(2)+" "+str(S_pick_time_tmp.second).zfill(2)+"."+str(S_pick_time_tmp.microsecond)+" GAU 1.00e-02  0.00e+00  0.00e+00  0.00e+00  1.00e+00") # DYJS   ?    ?    ? P      D 20140823 0534 0.3500 GAU 1.00e-01  0.00e+00  0.00e+00  0.00e+00  1.00e+00 
                    outfile.write("\n")
                except:
                    continue
                
                # Plot data for parameter setting:
                # if statName == "SKR01":
                #     fig, axes = plt.subplots(3)
                #     axes[0].plot(range(len(tr_filt.data)),tr_filt.data)
                #     axes[0].plot(range(len(tr_filt.data)),tr_filt.data)
                #     axes[0].axvline(x=(S_pick_secs_after_trace_start*tr.stats.sampling_rate), ymin=0.25, ymax=0.75)
                #     axes[1].plot(range(len(char_fnct_STA_LTA_array)), char_fnct_STA_LTA_array)
                #     tr_filt.spectrogram(axes=axes[2],wlen=0.02)
                #     plt.ylabel('Amplitude')
                #     plt.xlabel('Time [s]')
                #     plt.show()
   
            outfile.close()
        
    
def run(mseed_fname, outdir, stations_to_use, fract_stations_filter, freq_range=[45,200], min_snr=2.5, centre_freq_range=[4.0,200.0], centre_freq_range_step=0.5, band_width_gau_filter=[1.25], plotSwitch=False, dispersionFilterSwitch=True, returnCutMSEEDSwitch=False, verbosity_level=1):
    """Function to run spectrum based detection algorithm. Provides optional filtering for dispersive arrivals (rejecting arrivals exhibiting dispersive energy).
    Inputs are:
    Compulsary arguments:
    mseed_fname - Filename/s of set of mseed filenames to use. Must be single string (e.g. "20140629*SKR*.m"). Supports wildcards.
    outdir - Directory to save outputs to
    stations_to_use - List of stations to use (e.g. ["SKR01","SKR02","SKR03","SKR04","SKR05", "SKR06", "SKR07", "SKG08", "SKG10", "SKG11", "SKG12", "SKG13"])
    fract_stations_filter - Minimum number of stations that have to detect an arrival before triggering an event detection
    Optional arguments:
    freq_range - Range of frequencies over which to calculate the average energy used as the detection parameter (e.g. [45,200])
    min_snr - The minimum signal to noise ratio of the energy arriving within freq. band freq_range for triggering a detection
    centre_freq_range - Range of centre frequencies over which to do FTAN dispersion filtering (Hz) (e.g. [4.0,200.0])
    centre_freq_range_step - Size of centre frequency step used in dispersion filter (Hz) (e.g. 0.5)
    band_width_gau_filter - Band width of the gaussian filter applied during the dispersion filtering (Hz) (e.g. [1.25])
    plotSwitch - Boolian defining whether or not to produce key plots to check processing steps
    dispersionFilterSwitch - Boolian defining whether to implement disperison filtering or not
    returnCutMSEEDSwitch - Boolian defining whether or not to return cut mseed data for each event
    verbosity_level - Level of information given while running. (0=none, 1=key information, 2=all information)
    Outputs are:
    Detected events written to files and output to outdir
    Notes:
    mseed must have components specified. (i.e. must have Z,N,E specified in trace stats)
    Currently only does 1 second detection windows.
    """
    
    # Get stream for current data:
    st = get_single_mseed_file_data(mseed_fname)
    
    # Check that stations are definitely in stream, and if not, remove:
    if verbosity_level==1:
        print "---------------Checking that instruments are definitely in stream and list---------------"
    statName_list = stations_to_use
    statName_list_filtered_by_availability = []
    for j in range(len(st)):
        if str(st[j].stats.station) in statName_list:
            if str(st[j].stats.station) not in statName_list_filtered_by_availability:
                statName_list_filtered_by_availability.append(str(st[j].stats.station))
    statName_list = sorted(statName_list_filtered_by_availability)
    if verbosity_level==1:
        print "---------------Finished checking that instruments are definitely in stream and list---------------"


    # Perform search for icequakes from spectrogram:
    # Detect icequakes:
    if verbosity_level==1:
        print "---------------Performing initial spectrogram search for icequakes---------------"
    detection_all_stations_bins = icequake_search_using_spectrogram(st, statName_list, freq_range, min_snr, verbosity_level) # Returns proportion of station detections for each second in hour of data

    # And allow for overlap of events over multiple seconds:
    detection_all_stations_bins = adjust_for_event_second_overlap(detection_all_stations_bins) # Returns proportion of stations detections in each second, with any detections within the following second moved to the previous second
    if verbosity_level==1:
        print "---------------Finished initial spectrogram search for icequakes---------------"
    
    if verbosity_level==1:
        try:
            print "Number of events found within the data:", len(np.where(detection_all_stations_bins >= fract_stations_filter)[0])
        except:
            print "   "
        
    # Filter out dispersion events (surface waves from crevassing) (FTAN method):
    if dispersionFilterSwitch:
        if verbosity_level==1:
            print "---------------Performing surface wave filtering (via FTAN dispersion measurement) for icequakes---------------"
        sampling_rate = st[0].stats.sampling_rate # Sampling rate of signal in Hz
        # Get updated array using FTAN dispersion filter:
        detection_all_stations_bins = FTAN_filter_for_dispersion(st, detection_all_stations_bins, centre_freq_range, centre_freq_range_step, band_width_gau_filter, sampling_rate, statName_list, fract_stations_filter, plotSwitch, verbosity_level)
        if verbosity_level==1:
            print "---------------Finished surface wave filtering for icequakes---------------"

        if verbosity_level==1:
            try:
                print "Number of events found within the data, after filtering:", len(np.where(detection_all_stations_bins >= fract_stations_filter)[0])
            except:
                print "   "
            
    # Write events to file that have sufficient fraction of stations contributing to detection:
    if verbosity_level==1:
        print "---------------Writing event detection times to file---------------"
    # Write CMM event line data to file:
    produce_CMM_event_output_files(st, detection_all_stations_bins, fract_stations_filter, outdir, verbosity_level)
    # And create cut mseed data:
    if returnCutMSEEDSwitch:
        cut_mseed_for_events(st, detection_all_stations_bins, fract_stations_filter, outdir, verbosity_level)
    if verbosity_level==1:
        print "---------------Finished writing event detection times to file---------------"
    # Detect events that have sufficient fraction of stations contributing to detection, and get more precise arrival times:
    ###print "---------------Performing STA LTA detection of event arrivals for filterered icequakes, and writing to file---------------"
    # Write nonlinloc obs file data to file:
    ###STA_LTA_detect_event_arrivals_from_spectrogram_station_detection_fract(st, statName_list, detection_all_stations_bins, fract_stations_filter, outdir)
    ###print "---------------Finished STA LTA detection of event arrivals for filterered icequakes, and writing to file---------------"

# -------------------------------End of defining Main Functions In Script-------------------------------
 
    
# --------------------- Run functions in script: ---------------------
if __name__ == "__main__":
    # If run as main script run:
    run(jul_day, hour, datadir, outdir, stations_to_use, fract_stations_filter, freq_range=[45,200], min_snr=2.5, centre_freq_range=[4.0,200.0], centre_freq_range_step=0.5, band_width_gau_filter=[1.25], plotSwitch=False, dispersionFilterSwitch=True, verbosity_level=2)

