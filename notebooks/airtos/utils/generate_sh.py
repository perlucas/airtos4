# Use this Python script to automatically generate the .sh scripts that will guide the execution of the AI algorithm for each input set

# Input variables, each execution is a combination of these

LRATES = [0.003, 0.0044, 0.0058, 0.0072, 0.0086]

ALGORITHMS = ['macd', 'rsi', 'adx', 'mas']

LAYERS = ['v1', 'v2', 'v3', 'v4']

NUM_ITERATIONS = 35000000

SCRIPT_NAME = 'run_ppo.py'

OUTPUT_FILE = 'train.sh'

# Start file contents
file_contents = """# !bin/sh

echo 'Will start to run through each execution'
"""

total_executions = len(LRATES) * len(ALGORITHMS) * len(LAYERS)

file_contents += f"\necho 'Total executions found: {total_executions}'"
file_contents += "\necho ''"

cont = 1
for lrate in LRATES:
    for alg in ALGORITHMS:
        for layer in LAYERS:
            file_contents += f"""
echo 'Running execution {cont} of {total_executions}'
../bin/python {SCRIPT_NAME} NUMIT={NUM_ITERATIONS} LRATE={lrate} LAYERS={layer} ENV={alg} >> ./logs/run_{cont}.log
echo 'Finished execution'
"""
            cont+=1

with open(OUTPUT_FILE, "w") as file:
    # Writing data to a file
    file.write(file_contents)
