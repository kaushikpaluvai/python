import signal
import logging

def handle_timeout(signum, frame):
    print('Operation timed out')
    pass


class timeout:
    def __init__(self, seconds=1):
        self.seconds = seconds

    def __enter__(self):
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(self.seconds)
        else:
            logging.error("Cross validation timeout code not support on windows, code will not timeout.")

    def __exit__(self, type, value, traceback):
        if hasattr(signal, "alarm"):
            signal.alarm(0)
        else:
            logging.error("Cross validation timeout code not support on windows, code did not timeout.")
