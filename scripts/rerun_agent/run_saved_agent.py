import os

from tools.mnist_dataloaders import get_dataloaders, get_feature_dataloaders, reset_dataloader_batch_sizes
from models.models import model_name_to_function_dict
from models.wrapped_models import WrappedLenet5
from tools.agent_training import dqn_agent_training, dqn_agent_eval_offline, random_agent_eval
from tools.helper import copy_files_by_extension, set_random_seed, create_parser_from_yaml, split_network


def run_saved_agent(args):
    if args.set_seed:
        set_random_seed(args.seed)

    # copy_files_by_extension(args.source_folder, args.destination_folder, args.run_id, extension='zip')

    predictor_network_fn = model_name_to_function_dict[args.model]
    predictor_network = predictor_network_fn(**args.model_kwargs)
    _, predictor_head = split_network(network=predictor_network)

    datasets = get_dataloaders(args.dataset_task,
                               feature_extracted=True,
                               train_val_split=args.train_val_split,
                               seed=args.dataset_seed)

    wrapped_predictor = WrappedLenet5(model=predictor_network,
                                      criterion=datasets[0][2],
                                      tqdm_bar_enable=False,
                                      **args.wrapped_model_kwargs)
    if args.eval_rl_agent:
        for agent in args.agent_types:
            best_agent = True if agent == 'best_agent' else False
            queried_dataset_al_agent = dqn_agent_eval_offline(env_type=args.env_type,
                                                              wrapped_predictor=wrapped_predictor,
                                                              datasets_eval=datasets[0], al_budget=args.al_budget,
                                                              best_agent=best_agent,
                                                              log_folder_name=args.source_folder,
                                                              run_id=args.run_id, episode_count=1,
                                                              wandb_logging=args.wandb_logging,
                                                              wandb_project_name=args.wandb_project_name, args=args)

    if args.eval_random_agent:
        queried_dataset_random_agent = random_agent_eval(env_type=args.env_type,
                                                         wrapped_predictor=wrapped_predictor,
                                                         datasets_eval=datasets[0],
                                                         al_budget=args.al_budget,
                                                         log_folder_name=args.source_folder,
                                                         compare_with_agent=True,
                                                         run_name=args.dataset_task,
                                                         episode_count=1,
                                                         wandb_logging=args.wandb_logging,
                                                         wandb_project_name=args.wandb_project_name,
                                                         args=args)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    args = create_parser_from_yaml(default='agent_rerun', path=current_dir)
    run_saved_agent(args)
