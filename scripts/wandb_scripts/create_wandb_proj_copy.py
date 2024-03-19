import json
import wandb

# Set your API key
wandb.login()

# Set the source and destination projects

src_project = "sr2_results"
dst_project = "sr4_results"

# Initialize the wandb API
api = wandb.Api()

# Get the runs from the source project
runs = api.runs(f"{src_project}")

# Iterate through the runs and copy them to the destination project

for run in runs:
    # if run in runs_new_prj:
    #     continue
    # Get the run history and files
    history = run.history()
    # files = run.files()
    run_name = run.name

    if run_name.lower() == 'dqn_kmnist_400' or run_name.lower() == 'dqn_mnist_400' or run_name.lower() == 'dqn_cifar_400':



    # if '28' not in run_name.lower():
    #     continue
    # if '11' in run_name.lower() or '19' in run_name.lower() or '20' in run_name.lower():
    #     pass
    # else:
    #     continue


    # dataset_task = json.loads(run.json_config)['dataset_task']['value']
    # if dataset_task not in ['MNIST', 'kMNIST']:
    #     continue

    # Create a new run in the destination project
        new_run = wandb.init(project=dst_project, config=run.config, name=run.name, resume="allow")

        # Log the history to the new run
        for index, row in history.iterrows():
            new_run.log(row.to_dict())

        # # Upload the files to the new run
        # for file in files:
        #     file.download(replace=True)
        #     new_run.save(file.name, policy="now")

        # Finish the new run
        new_run.finish()

pass
