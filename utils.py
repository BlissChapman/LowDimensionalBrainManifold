import glob
import numpy as np
import os
import scipy.io


def load_fmri_timeseries(subject_id, trial_id, atlas='desikan'):
    '''
    Returns a 2D matrix with dimensions (Channels, Time) of the fMRI timeseries for the provided subject and trial.
    '''
    path = 'data/fmri/'+atlas+'_atlas/'+subject_id+'/repos'+str(trial_id)+'_timeseries_regr_wgm_globmean_filt0009008.mat'
    if os.path.isfile(path):
        return scipy.io.loadmat(path)['timeseries']
    else:
        return None
    
    
def compute_fmri_connectomes(subject_id, trial_id, seconds_used_to_compute_connectome, atlas='desikan'):
    '''
    Returns a 3D matrix with dimensions (Time, ConnectivityAxis1, ConnectivityAxis2) of the
    fMRI connectomes for the provided subject, trial, and seconds_per_connectome.
    '''
    assert(seconds_used_to_compute_connectome >= 2)
    
    # Load raw fMRI timeseries
    fmri_timeseries = load_fmri_timeseries(subject_id, trial_id, atlas=atlas)
    if fmri_timeseries is None:
        return
    
    # Filter to only cortical fMRI data
    cortical_fmri_timeseries = fmri_timeseries[18:]

    # Compute connectomes
    num_seconds_per_connectome = 2
    num_timepoints_per_connectome = int(seconds_used_to_compute_connectome/num_seconds_per_connectome) 
    fmri_connectome_matrices_through_time = []

    for t in range(num_timepoints_per_connectome, cortical_fmri_timeseries.shape[1]):
        cortical_fmri_timeseries_window = cortical_fmri_timeseries[:, t-num_timepoints_per_connectome:t]
        matrix = np.corrcoef(cortical_fmri_timeseries_window)
        fmri_connectome_matrices_through_time.append(matrix)
        
    return np.array(fmri_connectome_matrices_through_time)


def load_eeg_connectome(subject_id, trial_id, frequency_band, atlas='desikan'):
    '''
    Connectivity matrices are computed in the same step as source localization using the imaginary coherence measure.
    The time windows are non-overlapping 2 second windows corresponding to the time window of each concurrent fMRI volume.
    Therefore, we have 300 time windows with one connectivity matrix for each window per EEG oscillation band. 

    NOTE: The combined eeg connectomes returned from this method do not necessarily line up with the 
    fmri timeseries data returned from load_combined_fmri_timeseries.

    Returns a 3D matrix with dimensions (Time, ConnectivityAxis1, ConnectivityAxis2) for all eeg subject trials concatenated along the time dimension.
    '''
    
    path = 'data/eeg/'+atlas+'_atlas/'+subject_id+'/repos'+str(trial_id)+'/conn_desi_cohi_time_eeg_'+frequency_band+'_'+subject_id+'_repos'+str(trial_id)+'.mat'
    if os.path.isfile(path):
        data = scipy.io.loadmat(path)['connEEG'+frequency_band]
        data = np.moveaxis(data, 2, 0) # Move time axis to be the first axis
        return data
    else:
        return None
    
    
def load_artifact_timepoints(subject_id, trial_id, include_bad_fmri_frames=True):
    '''
    Returns the volume/window time indexes for EEG timepoints with artifacts. 
    Artifacts are determined with semi-automatic techniques and subsequent
    careful visual evaluation of the EEG data.
    
    If include_bad_fmri_frames is True, load_artifact_timepoints returns the 
    volume/window time indexes for EEG timepoints with artifacts UNIONED with 
    the set of timepoints with fMRI frame wise displacement.
    
    Note that a conservative threshold is used when computing fMRI frame wise displacement,
    resulting in a LOT of loss for some subjects.
    '''
    if include_bad_fmri_frames:
        path = 'data/head_motion/brainstorm_rejected_EEGandFD26.mat'
        artifact_timepoints_data = scipy.io.loadmat(path)['rejected_eeg'][0][0][0][0]
    else:
        path = 'data/head_motion/brainstorm_rejected26.mat'
        artifact_timepoints_data = scipy.io.loadmat(path)['subject'][0]

    
    for subject_data in artifact_timepoints_data:
        
        subject = subject_data[0][0]
        if subject != subject_id:
            continue
        
        for trial_data in subject_data[1][0]:
            
            trial = trial_data[0][0]
            if trial != 'repos'+str(trial_id):
                continue
            
            return trial_data[3][0]
    
    return []


def load_all_connectome_types(subject_id, trial_id, 
                             atlas='desikan', 
                             seconds_used_to_compute_fmri_connectome=60,
                             filter_artifact_timepoints=True,
                             exclude_bad_fmri_frames=True):
    '''
    Loads all fMRI and EEG connectomes for the specified subject and trial id. 
    Removes EEG connectomes that do not have an fMRI connectome correspondence.
    Removes artifact timepoints if specified.
    '''

    # Attempt to load both fmri and eeg connectomes
    fmri_connectomes = compute_fmri_connectomes(subject_id, trial_id, seconds_used_to_compute_fmri_connectome, atlas=atlas)
    alpha_eeg_connectomes  = load_eeg_connectome(subject_id, trial_id, frequency_band='alpha', atlas=atlas)
    beta_eeg_connectomes   = load_eeg_connectome(subject_id, trial_id, frequency_band='beta', atlas=atlas)
    delta_eeg_connectomes  = load_eeg_connectome(subject_id, trial_id, frequency_band='delta', atlas=atlas)
    gamma_eeg_connectomes  = load_eeg_connectome(subject_id, trial_id, frequency_band='gamma', atlas=atlas)
    theta_eeg_connectomes  = load_eeg_connectome(subject_id, trial_id, frequency_band='theta', atlas=atlas)
    broad_eeg_connectomes  = load_eeg_connectome(subject_id, trial_id, frequency_band='broad', atlas=atlas)

    if fmri_connectomes is None or alpha_eeg_connectomes is None or beta_eeg_connectomes is None or delta_eeg_connectomes is None or gamma_eeg_connectomes is None or theta_eeg_connectomes is None or broad_eeg_connectomes is None:
        return None
    
    # Note that there are less fmri connectomes than EEG connectomes because some number of frames are used to 
    # compute the initial fmri connectome. Here all EEG connectomes that do not have a corresponding fmri
    # connectome are dropped:
    num_dropped_frames = alpha_eeg_connectomes.shape[0] - fmri_connectomes.shape[0]
    
    alpha_eeg_connectomes = alpha_eeg_connectomes[num_dropped_frames:]
    beta_eeg_connectomes = beta_eeg_connectomes[num_dropped_frames:]
    delta_eeg_connectomes = delta_eeg_connectomes[num_dropped_frames:]
    gamma_eeg_connectomes = gamma_eeg_connectomes[num_dropped_frames:]
    theta_eeg_connectomes = theta_eeg_connectomes[num_dropped_frames:]
    broad_eeg_connectomes = broad_eeg_connectomes[num_dropped_frames:]

    # Load artifact timepoint labels if necessary
    if filter_artifact_timepoints:
        
        artifact_timepoints = load_artifact_timepoints(subject_id, trial_id, 
                                                       include_bad_fmri_frames=exclude_bad_fmri_frames)
        if artifact_timepoints is None:
            return None
        
        # Filter any artifact timepoints that were already dropped
        artifact_timepoints = list(filter(lambda a: a >= num_dropped_frames, artifact_timepoints))
        
        # Shift artifact timepoints to start at 0
        artifact_timepoints = [at - num_dropped_frames for at in artifact_timepoints]
            
        # Drop all 'bad' connectomes
        fmri_connectomes = np.delete(fmri_connectomes, artifact_timepoints, axis=0)
        alpha_eeg_connectomes = np.delete(alpha_eeg_connectomes, artifact_timepoints, axis=0)
        beta_eeg_connectomes = np.delete(beta_eeg_connectomes, artifact_timepoints, axis=0)
        delta_eeg_connectomes = np.delete(delta_eeg_connectomes, artifact_timepoints, axis=0)
        gamma_eeg_connectomes = np.delete(gamma_eeg_connectomes, artifact_timepoints, axis=0)
        theta_eeg_connectomes = np.delete(theta_eeg_connectomes, artifact_timepoints, axis=0)
        broad_eeg_connectomes = np.delete(broad_eeg_connectomes, artifact_timepoints, axis=0)
    
    return {
        'fmri':fmri_connectomes,
        'alpha':alpha_eeg_connectomes,
        'beta':beta_eeg_connectomes,
        'delta':delta_eeg_connectomes,
        'gamma':gamma_eeg_connectomes,
        'theta':theta_eeg_connectomes,
        'broad':broad_eeg_connectomes,
    }

ALL_SUBJECT_IDS = set([pth.split('/')[-1] for pth in glob.glob('data/**/**/*')])
ALL_TRIAL_IDS = range(1, 4)


def load_connectomes(subject_ids, trial_ids,
                     atlas='desikan', 
                     seconds_used_to_compute_fmri_connectome=60,
                     filter_artifact_timepoints=True,
                     exclude_bad_fmri_frames=True):
    '''
    Loads all connectome types for all specified subject/trial ids and
    concatenates them in the time dimension.
    '''
    
    # Gather connectomes for all specified subjects and trials
    all_subjects_all_trials_connectomes = []

    for subject_id in subject_ids:
        for trial_id in trial_ids:
            connectomes = load_all_connectome_types(subject_id, trial_id,
                                                   atlas=atlas, 
                                                   seconds_used_to_compute_fmri_connectome=seconds_used_to_compute_fmri_connectome,
                                                   filter_artifact_timepoints=filter_artifact_timepoints,
                                                   exclude_bad_fmri_frames=exclude_bad_fmri_frames)
            if connectomes is not None:
                all_subjects_all_trials_connectomes.append(connectomes)
            
    # Smush separate connectomes into one mega time series
    all_subjects_all_trials_connectomes_smushed = all_subjects_all_trials_connectomes[0]
    for i in range(1, len(all_subjects_all_trials_connectomes)):
        connectomes = all_subjects_all_trials_connectomes[i]
        for k in connectomes:
            all_subjects_all_trials_connectomes_smushed[k] = np.concatenate([all_subjects_all_trials_connectomes_smushed[k], connectomes[k]], axis=0)
        
    return all_subjects_all_trials_connectomes_smushed