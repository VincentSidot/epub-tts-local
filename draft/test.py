import os
import sys
import threading
import time


def inhibit_output(fucntion):
    def wrapper(*args, **kwargs):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            with open(os.devnull, "w") as devnull:
                sys.stdout = devnull
                sys.stderr = devnull
                return fucntion(*args, **kwargs)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    return wrapper


@inhibit_output
def worker():
    time.sleep(2)
    print("Work completed in the thread.")


# Create the thread
thread = threading.Thread(target=worker)

# Start the thread
thread.start()

# This print statement will execute immediately after starting the thread
print("Thread has been started and is running in the background.")

# Wait for the thread to complete
thread.join()

# This print statement will execute after the thread has completed
print("Thread has finished.")
