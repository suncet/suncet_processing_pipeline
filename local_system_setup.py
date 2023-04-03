import os
import setup_minimum_required_folders_files


# Setup steps
# 1. Manual: If you're contributing as part of the SunCET team, you should be part of the GitHub Organization -- talk to James Mason

# 2. Manual: This script assumes that you have already cloned the repository to some local directory, so do that.

# 3. Scripted: Install the environment using conda (you can do this manually with virtualenv instead if you prefer) -- this creates an environment called "suncet"
os.system('conda env create -f environment.yml')

# 4. Manual: Define environment variable for "suncet_data" path
suncet_data_path = input('What path do you want `suncet_data` to be? This is where the minimum set of required files to run the simulator will be downloaded: ')
os.environ['suncet_data'] = suncet_data_path
print('You are executing from the following shell:')
os.system('echo $SHELL')
print('HEY USER: Add an environment variable called suncet_data to your corresponding profile (e.g., ~/.zshrc, ~/.bash_profile, ~/.cshrc) that points to the path you just provided.' )

# 5. Scripted: Configure the necessary directory structure and download the minimum set of required files
setup_minimum_required_folders_files.run()

# 6. Scripted: Configure things so that pytest is always looking at the most recent edited code
os.system('pip install -e .')

# 7. Manual: In your dev tools (VS Code or PyCharm or whatever), open up the folder that you cloned

# 8. Manual: Setup your interpreter (e.g., in VS Code, open the Command Pallette, search for + click "Python: Select Interpreter", and choose the "suncet" environment that should've been installed in Step 3 above.)

# And then you're done! But you probably want to check that things are working. The quick way to do that is to open up a terminal in the folder of your local version of the repo, make sure you're in the "suncet" environment, and then just type "pytest". It should run and all tests should pass. 

# For running, developing, and debugging the code, make sure to run things from the main wrapper: simulator.py. That calls the other key files/classes/functions (e.g., instrument.py). 
# If you're in an IDE, the typical process would be:
# 1. Make edits in whatever file you're working on, and put a break point there
# 2. Go to simulator.py and tell the IDE to run it in debug mode
# 3. Code will execute until it reaches your breakpoint, at which point you can inspect variables, test your code, and test alternative implementations of code at the command line in the debugger
# 4. Make any needed changes to your code, and repeat this process