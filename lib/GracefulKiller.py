import signal
import time

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True

    def wait(self):
        while not self.kill_now:
            time.sleep(1)

if __name__ == "__main__":
    print("start")
    gk = GracefulKiller()
    gk.wait()
    print("stop")