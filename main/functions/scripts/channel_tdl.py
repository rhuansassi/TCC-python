import numpy as np

def channel_tdl(ts, delay_spread, channel):
    """
    Compute rayleigh gains over time. If t_init != 0, the channel will present
    a discontinuity compared with t_init-ts since they possibly consider
    different normalization factors to keep unitary power!

    Parameters:
        :ts (float): sampling period.
        :f_doppler (float): doppler frequency.
        :n_tx (int): number of transmitters.
        :n_rx (int): number of receivers.
        :gains (numpy.ndarray): Average path gains [dB].
        :k_paths (float): Number of paths to compute.
        :n_samples (int): Number of channel samples to compute.
        :m_scatters (int): Number of scatters computed for Doppler shift.
        :t_init (float): Initial time for running.
        :seed (int): Seed of random numbers to guarantee reproducibility.

    Returns:
        :paths (numpy.ndarray): obtained paths (complex128).
    """

    # Initialize dictionary of channel parameters
    channel_dic = {}

    # For TDL-A channel
    if channel == 'TDL-A':
        n_delays,gains_db,los,k_factor,los_power,mean_los_power = __tdl_a()
    # For TDL-B channel
    elif channel == 'TDL-B':
        n_delays,gains_db,los,k_factor,los_power,mean_los_power = __tdl_b()
    # For TDL-C channel
    elif channel == 'TDL-C':
        n_delays,gains_db,los,k_factor,los_power,mean_los_power = __tdl_c()
    # For TDL-D channel
    elif channel == 'TDL-D':
        n_delays,gains_db,los,k_factor,los_power,mean_los_power = __tdl_d()
    # For TDL-E channel
    elif channel == 'TDL-E':
        n_delays,gains_db,los,k_factor,los_power,mean_los_power = __tdl_e()

    # Time delays scaled by the delay spread
    t_delays = n_delays*delay_spread

    # Sampled delays
    s_delays = round(t_delays/ts)

    # Unique delays
    u_delays = unique(s_delays).astype(int)

    # Convert gains to linear
    lin_gains = 10**(gains_db/20)

    # Join gains
    joined_gains = array([sum(lin_gains[argwhere(s_delays == u_d)])
                          for u_d in u_delays])

    # Gains power
    g_power = sum(abs(joined_gains)**2)/joined_gains.shape[0]

    # Normalize gain power to one
    gains_power = joined_gains/sqrt(g_power)

    # Get parameters
    channel_dic['gains']          = gains_power
    channel_dic['delays']         = u_delays
    channel_dic['los']            = los
    channel_dic['k_factor']       = 10**(k_factor/20)
    channel_dic['los_power']      = 10**(los_power/20)
    channel_dic['mean_los_power'] = 10**(mean_los_power/20)

    return channel_dic
# *****************************************************************************
# *****************************************************************************

# *****************************************************************************
# *****************************************************************************
def __tdl_a():
    """
    TDL-A channel delays and gains

    Parameters:

    Returns:
        :delays (numpy.ndarray): normalized delays.
        :paths (numpy.ndarray): gains in dB.
    """

    # time delays
    delays= np.array([0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708,
                  0.5750, 0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942,
                  2.5119, 3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066,
                  5.3043, 9.6586])

    # gains
    gains= np.array([-13.4,   0.0,  -2.2,  -4.0,  -6.0,  -8.2,  -9.9, -10.5,  -7.5,
                 -15.9,  -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2,
                 -18.3, -18.9, -16.6, -19.9, -29.7])

    return delays, gains, False, 1, 1, 1
# *****************************************************************************
# *****************************************************************************

# *****************************************************************************
# *****************************************************************************
def __tdl_b():
    """
    TDL-B channel delays and gains

    Parameters:

    Returns:
        :delays (numpy.ndarray): normalized delays.
        :paths (numpy.ndarray): gains in dB.
    """

    # time delays
    delays=array([0.0000, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752,
                  0.5055, 0.3681, 0.3697, 0.5700, 0.5283, 1.1021, 1.2756,
                  1.5474, 1.7842, 2.0169, 2.8294, 3.0219, 3.6187, 4.1067,
                  4.2790, 4.7834])

    # gains
    gains=array([ 0.0, -2.2, -4.0, -3.2, -9.8, -1.2, -3.4,  -5.2, -7.6,  -3.0,
                 -8.9, -9.0, -4.8, -5.7, -7.5, -1.9, -7.6, -12.2, -9.8, -11.4,
                -14.9, -9.2,-11.3])

    return delays, gains, False, 1, 1, 1
# *****************************************************************************
# *****************************************************************************

# *****************************************************************************
# *****************************************************************************
def __tdl_c():
    """
    TDL-C channel delays and gains

    Parameters:

    Returns:
        :delays (numpy.ndarray): normalized delays.
        :paths (numpy.ndarray): gains in dB.
    """

    # time delays
    delays=array([0.0000, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448,
                  0.6560, 0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083,
                  2.1704, 2.7105, 4.2589, 4.6003, 5.4902, 5.6077, 6.3065,
                  6.6374, 7.0427])

    # gains
    gains=array([ -4.4,  -1.2,   -3.5,  -5.2,  -2.5,   0.0,  -2.2,  -3.9, -7.4,
                  -7.1, -10.7 , -11.1,  -5.1,  -6.8,  -8.7, -13.2, -13.9,-13.9,
                 -15.8, -17.1,  -16.0, -15.7, -21.6, -22.8])

    return delays, gains, False, 1, 1, 1
# *****************************************************************************
# *****************************************************************************

# *****************************************************************************
# *****************************************************************************
def __tdl_d():
    """
    TDL-D channel delays and gains

    Parameters:

    Returns:
        :delays (numpy.ndarray): normalized delays.
        :paths (numpy.ndarray): gains in dB.
    """

    # time delays
    delays=array([0.0000, 0.0350, 0.6120, 1.3630, 1.4050, 1.7750, 1.8040,
                  2.5960, 4.0420, 7.9370, 9.4240, 9.7080, 12.525])

    # gains
    gains=array([-13.5, -18.8, -21.0, -22.8, -17.9, -22.9, -20.1, -21.9, -27.8,
                 -23.6, -24.8, -30.0, -27.7])

    return delays, gains, True,  13.3, -0.2, 0
# *****************************************************************************
# *****************************************************************************

# *****************************************************************************
# *****************************************************************************
def __tdl_e():
    """
    TDL-E channel delays and gains

    Parameters:

    Returns:
        :delays (numpy.ndarray): normalized delays.
        :paths (numpy.ndarray): gains in dB.
    """

    # time delays
    delays=array([0.0000, 0.5133, 0.5440, 0.5630, 0.5440, 0.71120, 1.9092,
                  1.9293, 1.9589, 2.6426, 3.7136, 5.4524, 12.0034, 20.6519])

    # gains
    gains=array([-22.03, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6, -20.8,
                 -22.60, -22.3, -25.6, -20.2, -29.8, -29.2])

    return delays, gains, True,  22, -0.03, 0
# *****************************************************************************
# *****************************************************************************