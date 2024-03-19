import re
import wandb

# Set your API key
wandb.login()

# Set the source and destination projects

src_project = "sr3_results"

# Initialize the wandb API
api = wandb.Api()

# Get the runs from the source project
runs = api.runs(f"{src_project}")

# Iterate through the runs and copy them to the destination project

for run in runs:
    # if run in runs_new_prj:
    #     continue
    # Get the run history and files

    # files = run.files()
    og_name = run.name
    if og_name == 'DQN_kmnist_400':
        run.name = 'DQN_kmnist_400_sr1'
        run.update()
    # last_underscore_index = og_name.rfind('_')
    # if re.search(r'\d', og_name[last_underscore_index:]):
    #     # Slice the string to exclude the part after the last underscore
    #     new_name = og_name[:last_underscore_index]
    #     run.name = new_name
    #     run.update()

    # run.finish()
