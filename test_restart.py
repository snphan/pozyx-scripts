from subprocess import run
from time import sleep

# Path and name to the script you are trying to start
file_path = "multitag_position copy.py" 


restart_timer = 2
def start_script():
    try:
        # Make sure 'python' command is available
        run("python3.10 "+file_path, check=True) 
    except:
        # Script crashed, lets restart it!
        handle_crash()

def handle_crash():
    sleep(restart_timer)  # Restarts the script after 2 seconds
    print('Restarting')
    start_script()

start_script()