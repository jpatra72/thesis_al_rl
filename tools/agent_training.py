import os
import random
import time
from typing import Optional, Tuple, Dict, Union
from argparse import Namespace

import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

import custom_envs
from models.wrapped_models import BaseNetworkWrapper
from tools.custom_callbacks import TensorboardCallback
from tools.helper import DEVICE, get_logs_subfolder_path, get_run_log_path, rename_folder, wandb_folder_path

env_name_dict = {'standard': 'custom_envs/ActiveImageClassifier-v1',
                 }
dqn_policy_dict = {'standard': 'MlpPolicy',
                   }
policy_kwargs_dict = {'standard': dict(net_arch=[128, 128]),
                      }


def dqn_agent_training(
        env_type: str,
        wrapped_predictor: BaseNetworkWrapper,
        datasets: Tuple[Optional[int], Tuple[Dataset, Dataset, Dataset], nn.Module],
        log_folder_name: str,
        eval_frequency: int,
        args: Namespace,
        wandb_project_name: str,
        wandb_logging: bool = True,
        n_env_train: int = 2,
        al_budget: int = 100,
        test_pipeline: bool = False,
        env_seed: Optional[int] = None,
) -> (DQN, str):
    n_env = [n_env_train, 1] if args.eval_during_training else [n_env_train]

    env_list = [gym.make(id=env_name_dict[env_type],
                         model=wrapped_predictor,
                         dataset=datasets[1][idx],
                         test_dataset=datasets[1][2],
                         al_budget=al_budget,
                         state_type=args.state_type,
                         eval_env=False if idx == 0 else True
                         ) for idx, _ in enumerate(n_env)]

    logs_folder_path = get_logs_subfolder_path(log_folder_name)

    model = DQN(dqn_policy_dict[env_type],
                env_list[0],
                buffer_size=args.buffer_size,
                learning_rate=args.learning_rate,
                learning_starts=args.learning_starts,
                gamma=args.gamma,
                train_freq=args.train_freq,
                gradient_steps=args.gradient_steps,
                batch_size=args.batch_size,
                exploration_fraction=args.exploration_fraction,
                exploration_initial_eps=args.exploration_initial_eps,
                exploration_final_eps=args.exploration_final_eps,
                target_update_interval=args.target_update_interval,
                policy_kwargs=policy_kwargs_dict[env_type],
                device=DEVICE,
                verbose=1,
                tensorboard_log=logs_folder_path,
                # seed=args.seed,
                )

    run_log_path, run_name = get_run_log_path(logs_folder_path, run_name_prefix='DQN')
    wandb_run_name = f"{run_name}_test" if test_pipeline else run_name
    if wandb_logging:
        wandb.init(project=wandb_project_name, name=wandb_run_name, config=vars(args), sync_tensorboard=True,
                   dir=wandb_folder_path)

    callbacks = [TensorboardCallback()]

    if args.eval_during_training:
        # eval AL agent using the same split used to train AL agent
        eval_callback = EvalCallback(env_list[1],
                                     best_model_save_path=run_log_path,
                                     log_path=run_log_path,
                                     eval_freq=eval_frequency)
        callbacks.append(eval_callback)

    model.learn(total_timesteps=args.total_timesteps,
                log_interval=1,
                callback=callbacks)
    model.save(os.path.join(run_log_path, 'final_model'))
    model.save_replay_buffer(os.path.join(run_log_path, 'replay_buffer'))

    if wandb_logging:
        wandb.finish()
    if test_pipeline:
        _, run_name = rename_folder(logs_folder_path, run_name, wandb_run_name)
    dqn_run_id = run_name.split('_', 1)[-1]
    return model, dqn_run_id


def dqn_agent_eval_offline(env_type: str, wrapped_predictor: BaseNetworkWrapper,
                           datasets_eval: Tuple[Optional[int], Tuple[Dataset, Dataset, Dataset], nn.Module],
                           al_budget: int, best_agent: bool, log_folder_name: str, run_id: str, episode_count: int,
                           wandb_logging: bool, wandb_project_name: str, args: Namespace) -> Dataset:
    logs_folder_path = get_logs_subfolder_path(log_folder_name)
    run_name = f'DQN_{run_id}'
    agent_filename = 'best_model' if best_agent else 'final_model'
    trained_agent_path = os.path.join(logs_folder_path, run_name, agent_filename)

    if wandb_logging:
        wandb.init(project=wandb_project_name, name=f'{run_name}_{args.dataset_task}_{args.al_budget}', sync_tensorboard=True,
                   config=vars(args), dir=wandb_folder_path)
    subfolder_path = os.path.join(logs_folder_path, run_name, f"{int(random.random() * 1000)}")
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    writer = SummaryWriter(log_dir=subfolder_path)

    eval_env = gym.make(id=env_name_dict[env_type],
                        model=wrapped_predictor,
                        dataset=datasets_eval[1][1],
                        test_dataset=datasets_eval[1][2],
                        al_budget=al_budget,
                        state_type=args.state_type,)

    agent = DQN.load(trained_agent_path)

    queried_dataset = run_episodes(eval_env,
                                   episode_count=episode_count,
                                   agent=agent,
                                   writer=writer,
                                   best_agent=best_agent)
    if wandb_logging:
        wandb.finish()
    return queried_dataset


def dqn_agent_eval_online(env_type: str, wrapped_predictor: BaseNetworkWrapper,
                          datasets_eval: Tuple[Optional[int], Tuple[Dataset, Dataset, Dataset], nn.Module],
                          al_budget: int, best_agent: bool, log_folder_name: str, run_id: str, episode_count: int,
                          wandb_logging: bool, wandb_project_name: str, args: Namespace) -> Dataset:
    logs_folder_path = get_logs_subfolder_path(log_folder_name)
    run_name = f'DQN_{run_id}'
    agent_filename = 'best_model' if best_agent else 'final_model'
    trained_agent_path = os.path.join(logs_folder_path, run_name, agent_filename)
    replay_buffer_path = os.path.join(logs_folder_path, run_name, 'replay_buffer')
    if wandb_logging:
        wandb.init(project=wandb_project_name, name=run_name, sync_tensorboard=True,
                   config=vars(args), dir=wandb_folder_path)
    writer = SummaryWriter(log_dir=os.path.join(logs_folder_path, run_name))

    eval_env = gym.make(id=env_name_dict[env_type],
                        model=wrapped_predictor,
                        dataset=datasets_eval[1][1],
                        test_dataset=datasets_eval[1][2],
                        al_budget=al_budget,
                        sort_similarity=args.sort_similarity,)
    eval_env = DummyVecEnv([lambda: eval_env])

    custom_objs = {'tensorboard_log': '/Users/washington/Desktop/3.5 DLR/Github/ALPBNN/logs/cl_runs_small_scale_exp',
                   }
    new_logger = configure(logs_folder_path, ["stdout"])
    agent = DQN.load(trained_agent_path, env=eval_env, custom_objects=custom_objs, device=DEVICE)
    # agent = DQN.load(trained_agent_path, env=eval_env)
    agent.load_replay_buffer(replay_buffer_path)
    agent.set_logger(new_logger)
    agent.learn(0, reset_num_timesteps=True)
    agent.replay_buffer.device = DEVICE

    queried_dataset = run_episodes_online(agent=agent,
                                          agent_train_kwargs=args.dqn_train_kwargs,
                                          env=eval_env,
                                          episode_count=episode_count,
                                          writer=writer,
                                          best_agent=best_agent)
    if wandb_logging:
        wandb.finish()
    return queried_dataset


def random_agent_eval(
        env_type: str,
        wrapped_predictor: BaseNetworkWrapper,
        datasets_eval: Tuple[Optional[int], Tuple[Dataset, Dataset, Dataset], nn.Module],
        al_budget: int,
        log_folder_name: str,
        compare_with_agent: bool,
        run_name: str,
        episode_count: int,
        wandb_logging: bool,
        wandb_project_name: str,
        args: Namespace,
) -> Dataset:
    logs_folder_path = get_logs_subfolder_path(log_folder_name)
    # run_name = f'DQN_{run_id}_random' if compare_with_agent else run_id

    if wandb_logging:
        # resume = True if compare_with_agent else False
        wandb.init(project=wandb_project_name, name=f"random_{run_name}_{args.al_budget}", sync_tensorboard=True,
                   dir=wandb_folder_path, config=vars(args))
    writer = SummaryWriter(log_dir=os.path.join(logs_folder_path, f'{run_name}_random_{args.seed}'))

    eval_env = gym.make(id=env_name_dict[env_type],
                        model=wrapped_predictor,
                        dataset=datasets_eval[1][1],
                        test_dataset=datasets_eval[1][2],
                        al_budget=al_budget,
                        sort_similarity=args.sort_similarity,
                        )

    queried_dataset = run_episodes(eval_env,
                                   episode_count=episode_count,
                                   agent=None,
                                   writer=writer)
    if wandb_logging:
        wandb.finish()
    return queried_dataset


def run_episodes(
        env: gym.Env,
        episode_count: int,
        writer: SummaryWriter,
        best_agent: Optional[bool] = None,
        agent: Optional[DQN] = None,
) -> Dataset:
    if agent is None:
        tag_suffix_list = ['best_model', 'final_model']
    else:
        tag_suffix_list = ['best_model'] if best_agent else ['final_model']
    eps_count = 0
    env_step_count = 0
    queried_count = 0
    obs, info = env.reset()
    while eps_count < episode_count:
        if agent:
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = np.random.randint(2)
        obs, reward, done, _, info = env.step(action)
        env_step_count += 1
        if action and writer is not None:
            for tag_suffix in tag_suffix_list:
                writer.add_scalar(f'infer_{tag_suffix}/model_performance', info['model_performance']['test_data'],
                                  global_step=queried_count)
                writer.add_scalar(f'infer_{tag_suffix}/model_entropy_test_data', info['model_entropy']['test_data'],
                                  global_step=queried_count)
                # writer.add_scalar(f'infer_{tag_suffix}/model_entropy_test_data_mov_avg',
                #                   info['model_entropy']['test_data_eps_avg'],
                #                   global_step=queried_count)
                writer.add_scalar(f'infer_{tag_suffix}/step_count', env_step_count, global_step=queried_count)
            queried_count += 1

        if done:
            queried_dataset = env.unwrapped.dataloader_labelled.dataset
            # added small random noise to make histogram logging work with weights & biases
            class_labels = [[i, label + random.random() / 100] for i, label in enumerate(queried_dataset.dataset.tensors[1][queried_dataset.indices])]
            # table = wandb.Table(data=class_labels, columns=['index', 'Class Labels'])
            # histogram = wandb.plot.histogram(table, value='Class Labels', title='Histogram')
            # for tag_suffix in tag_suffix_list:
            #     wandb.log({f'infer_{tag_suffix}/class_labels': histogram})

            env_step_count = 0
            queried_count = 0
            print(f"Episode: {eps_count} complete!")
            obs, info = env.reset()
            eps_count += 1
    return queried_dataset


def run_episodes_online(
        agent: DQN,
        agent_train_kwargs: Dict,
        env: DummyVecEnv,
        episode_count: int,
        writer: SummaryWriter,
        best_agent: Optional[bool] = None,
) -> Dataset:
    tag_suffix_list = ['best_model'] if best_agent else ['final_model']
    eps_count = 0
    env_step_count = 0
    queried_count = 0
    obs = env.reset()
    while eps_count < episode_count:
        action, _ = agent.predict(obs, deterministic=True)
        next_obs, reward, done, info = env.step(action)
        env_step_count += 1
        agent.replay_buffer.add(obs, next_obs, action, reward, done, info)
        if env_step_count % 4 == 0:
            agent.train(**agent_train_kwargs)
        if action and writer is not None:
            for tag_suffix in tag_suffix_list:
                writer.add_scalar(f'infer_{tag_suffix}/model_performance', info[0]['model_performance'],
                                  global_step=queried_count)
                writer.add_scalar(f'infer_{tag_suffix}/model_entropy_test_data', info[0]['model_entropy']['test_data'],
                                  global_step=queried_count)
                writer.add_scalar(f'infer_{tag_suffix}/model_entropy_test_data_mov_avg',
                                  info[0]['model_entropy']['test_data_eps_avg'],
                                  global_step=queried_count)
                writer.add_scalar(f'infer_{tag_suffix}/step_count', env_step_count, global_step=queried_count)
            queried_count += 1

        if done:
            queried_dataset = env.envs[0].unwrapped.queried_dataset
            # added small random noise to make histogram logging work with weights & biases
            class_labels = [[i, label + random.random() / 100] for i, label in enumerate(queried_dataset.labels)]
            table = wandb.Table(data=class_labels, columns=['index', 'Class Labels'])
            histogram = wandb.plot.histogram(table, value='Class Labels', title='Histogram')
            for tag_suffix in tag_suffix_list:
                wandb.log({f'infer_{tag_suffix}/class_labels': histogram})

            env_step_count = 0
            queried_count = 0
            print(f"Episode: {eps_count} complete!")
            next_obs = env.reset()
            eps_count += 1
        obs = next_obs
    return queried_dataset
