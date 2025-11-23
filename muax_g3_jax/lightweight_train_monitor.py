import time
import math


class TrainMonitor:
    """
    Extremely lightweight training monitor.

    - begin_episode()
    - step(reward, **step_metrics)
    - record_metrics(dict_of_metrics)  # extra stuff like loss, training_step, test_G
    - end_episode()  # prints one log line
    """

    def __init__(self, smoothing: int = 10, name: str = "TrainMonitor"):
        self.smoothing = float(smoothing)
        self.name = name
        self.reset_global()

    def reset_global(self):
        self.T = 0              # global steps
        self.ep = 0             # episodes
        self.t = 0              # steps in current ep
        self.G = 0.0            # return in current ep
        self.avg_G = 0.0        # smoothed return
        self._n_avg_G = 0.0
        self._ep_start_time = time.time()
        self._last_metrics = {}
        self._in_episode = False

    @property
    def avg_r(self) -> float:
        if self.t == 0:
            return math.nan
        return self.G / self.t

    @property
    def dt_ms(self) -> float:
        if self.t == 0:
            return math.nan
        return 1000.0 * (time.time() - self._ep_start_time) / self.t

    def begin_episode(self):
        if self._in_episode:
            self.end_episode()

        self.ep += 1
        self.t = 0
        self.G = 0.0
        self._ep_start_time = time.time()
        self._last_metrics = {}
        self._in_episode = True

    def step(self, reward: float, **metrics):
        """Call this once per env step."""
        if not self._in_episode:
            self.begin_episode()

        self.T += 1
        self.t += 1
        self.G += float(reward)
        self._last_metrics.update(metrics)

    def record_metrics(self, metrics: dict):
        """Extra metrics not tied to a single env step (e.g. loss averaged over episode)."""
        self._last_metrics.update(metrics)

    def end_episode(self):
        if not self._in_episode:
            return
        self._in_episode = False

        # update running-average of return
        if self._n_avg_G < self.smoothing:
            self._n_avg_G += 1.0
        self.avg_G += (self.G - self.avg_G) / self._n_avg_G

        self._print_line()

    def _print_line(self):
        avg_r = self.avg_r
        dt = self.dt_ms

        v = float(self._last_metrics.get("v", math.nan))
        Rn = float(self._last_metrics.get("Rn", math.nan))
        loss = float(self._last_metrics.get("loss", math.nan))
        training_step = float(self._last_metrics.get("training_step", math.nan))

        msg = (
            f"[{self.name}|INFO] "
            f"ep: {self.ep},\t"
            f"T: {self.T:,},\t"
            f"G: {self.G:.0f},\t"
            f"avg_r: {avg_r:.3g},\t"
            f"avg_G: {self.avg_G:.1f},\t"
            f"t: {self.t},\t"
            f"dt: {dt:.3f}ms,\t"
            f"v: {v:.1f},\t"
            f"Rn: {Rn:.1f},\t"
            f"loss: {loss:.2f},\t"
            f"training_step: {training_step:.2e}"
        )
        print(msg)
