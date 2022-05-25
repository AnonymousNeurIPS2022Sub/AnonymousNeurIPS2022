import os
import time


class BasePlotter():
    def __init__(self, file, interval):
        self.file = file
        self.interval = interval

    def plot_now(self, r):
        # if r is anything else then an int we always want to plot it
        if type(r) is int and r % self.interval != 0:
            return False
        return True


class BeforeEpochPlotter(BasePlotter):
    def __init__(self, file, interval):
        super().__init__(file, interval)

    def plot_now(self, r, call_position):
        if 'before' in call_position:
            return super().plot_now(r)
        else:
            return False


class AfterEpochPlotter(BasePlotter):
    def __init__(self, file, interval):
        super().__init__(file, interval)

    def plot_now(self, r, call_position):
        if 'after' in call_position:
            return super().plot_now(r)
        else:
            return False
