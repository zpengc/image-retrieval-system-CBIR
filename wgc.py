import numpy as np
from utils import wrap_angle
from utils import wrap_log_scale


# weak geometry consistency
# axis=0 : operate on each column
# axis=1 : operate on each row
class WGC:

    # 2000, 17 ,7
    def __init__(self, n, angle_bins, scale_bins):
        self.n = n
        self.angle_bins = angle_bins
        self.scale_bins = scale_bins
        self.angle_histograms = np.zeros((self.n, self.angle_bins))
        self.scale_histograms = np.zeros((self.n, self.scale_bins))

    def vote(self, image_id, angle_diff, log_scale_diff):
        angle_index = wrap_angle(angle_diff)
        scale_index = wrap_log_scale(log_scale_diff)
        if 0 <= angle_index < self.angle_bins:
            self.angle_histograms[image_id][angle_index] += 1
        if 0 <= scale_index < self.scale_bins:
            self.scale_histograms[image_id][scale_index] += 1

    def filter(self):
        am = np.max([h for h in self.angle_histograms], axis=1)
        sm = np.max([h for h in self.scale_histograms], axis=1)
        # print(am)   # [54.         34.66666667 24.33333333 ... 20.33333333 15.11.        ]
        # print(sm)   # [58.66666667 44.33333333 37.33333333 ... 23.33333333 13.66666667 11.66666667]
        # print(am.shape)  # (2000,)
        # print(sm.shape)  # (2000,)
        # print(np.vstack((am, sm)).shape)    # (2,2000)
        return np.min(np.vstack((am, sm)), axis=0)  # (2000,)


