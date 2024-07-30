class waveform(object):
    def __init__(self, centre_freq, chirp_bandwidth, chirp_period, prf):
        self.centre_freq = centre_freq
        self.chirp_bandwidth = chirp_bandwidth
        self.chirp_period = chirp_period
        self.prf = prf
        self.range_bins = None
        