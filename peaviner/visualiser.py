from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

@dataclass
class PeaVis:
    ax: plt.Axes = None
    gamma: float = None

    def setup_axis(self, ax: plt.Axes, gamma: float, aspect=1):
        ticks = list(np.arange(0, gamma-0.025, 0.05)) + [gamma]
        ax.set_ylim(0, ticks[-1])
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:.2f}" for t in ticks])
        ax.set_ylabel('tp, true positive', loc='top')

        ticks = list(np.arange(0, 1-gamma-0.025, 0.05)) + [1-gamma]
        ax.set_xlim(0, ticks[-1])
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.2f}" for t in ticks])
        ax.set_xlabel('fp, false positive', loc='right')
        #for spine in ['right', 'top']:
        #    ax.spines[spine].set_visible(False)

        ax.set_aspect(aspect)

        self.ax = ax
        self.gamma = gamma

    def draw_thold(self, thold: float, score_name: str = 'Jaccard'):
        if score_name == 'Jaccard':
            thold_x = self.gamma*(1-thold)/thold
            thold_y = self.gamma*thold
        else:
            return NotImplementedError('Only Jaccard score is implemented at the moment')

        self.ax.plot([0, thold_x], [thold_y, self.gamma], label=r'$\theta: '+f"{thold:.2f}"+'$')

        return thold_x, thold_y

    def draw_zones(
            self, thold_x, thold_y,
            fill_zones=True, fill_alpha=0.2, zone_colors=('darkgreen', 'limegreen', 'yellow', 'orange', 'red'),
            caption_zones=True, caption_size=18,
    ):
        self.ax.axvline(thold_x, linestyle='--')
        self.ax.axhline(thold_y, linestyle='--')

        if fill_zones:
            self.ax.fill([thold_x, 1 - self.gamma, 1 - self.gamma, thold_x], [0, 0, thold_y, thold_y],
                         alpha=fill_alpha, color=zone_colors[4], zorder=0)
            self.ax.fill([thold_x, 1 - self.gamma, 1 - self.gamma, thold_x], [thold_y, thold_y, self.gamma, self.gamma],
                         alpha=fill_alpha, color=zone_colors[3], zorder=0)
            self.ax.fill([0, thold_x, thold_x, 0], [0, 0, thold_y, thold_y],
                         alpha=fill_alpha, color=zone_colors[2], zorder=0)
            self.ax.fill([0, thold_x, thold_x], [thold_y, thold_y, self.gamma],
                         alpha=fill_alpha, color=zone_colors[1], zorder=0)
            self.ax.fill([0, thold_x, 0], [thold_y, self.gamma, self.gamma],
                         alpha=fill_alpha, color=zone_colors[0], zorder=0)

        if caption_zones:
            kws = dict(size=caption_size, ha='right', va='bottom', weight='bold')
            self.ax.text(0 + 0.01, self.gamma - 0.01, 'I', **dict(kws, ha='left', va='top'))
            self.ax.text(thold_x - 0.01, thold_y + 0.01, 'II', **kws)
            self.ax.text(thold_x - 0.01, 0 + 0.01, 'III', **kws)
            self.ax.text(1 - self.gamma - 0.01, thold_y + 0.01, 'IV', **kws)
            self.ax.text(1 - self.gamma - 0.01, 0 + 0.01, 'V', **kws)
