from pathlib import Path
import matplotlib.pyplot as plt


def ensure_dir(path):
    path = Path(path)
    if not path:
        return False
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
    return path.is_dir()


def save_figure(path, fig=None, dpi=300, enabled=True):
    if not enabled:
        return
    path = Path(path)
    if fig is not None:
        # Make the figure with fig the current figure
        plt.figure(fig.number)
    if not ensure_dir(path.parent):
        assert False, "Failed to create output directory: %s " % path.parent
    plt.savefig(path, bbox_inches="tight", dpi=dpi)
