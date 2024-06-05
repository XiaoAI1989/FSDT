import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics
from hydra.utils import instantiate

import utils
from buffer import ClientBuffer
from model import DecisionTransformer
from train import train_global_client, train_transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def eval(env: gym.vector.Env, model: DecisionTransformer, rtg_target):
    # parallel evaluation with vectorized environment
    model.eval()
    
    episodes = env.num_envs
    reward, returns = np.zeros(episodes), np.zeros(episodes)
    done_flags = np.zeros(episodes, dtype=np.bool8)

    state_dim = utils.get_space_shape(env.observation_space, is_vector_env=True)
    act_dim = utils.get_space_shape(env.action_space, is_vector_env=True)
    max_timestep = model.max_timestep
    context_len = model.context_len
    # each vectorized environment us
    timesteps = torch.tile(torch.arange(max_timestep, device=device), (episodes, 1))

    state, _ = env.reset(seed=[np.random.randint(0, 10000) for _ in range(episodes)])
    
    # placeholder for states, actions, rewards_to_go
    states = torch.zeros((episodes, max_timestep, state_dim), dtype=torch.float32, device=device)
    actions = torch.zeros((episodes, max_timestep, act_dim), dtype=torch.float32, device=device)
    rewards_to_go = torch.zeros((episodes, max_timestep, 1), dtype=torch.float32, device=device)

    reward_to_go, timestep = rtg_target, 0

    while not done_flags.all():

        states[:, timestep] = torch.tensor(state)
        rewards_to_go[:, timestep] = torch.tensor(reward_to_go)
        timestep += 1
        if timestep - context_len <= 0:
            start = 0
        else:
            start = timestep - context_len
        state_preds, action_preds, reward_to_go_preds = model(states[:, start:timestep],
                                                              actions[:, start:timestep],
                                                              rewards_to_go[:, start:timestep],
                                                              timesteps[:, start:timestep])

        action = action_preds[:, -1]
        actions[:, timestep - 1] = action

        action = action.detach().cpu().numpy()
        state, reward, done, truncated, _ = env.step(action)

        done_flags = (1-(1 - done)*(1-truncated)*(1-done_flags))
        reward=reward * (1-done_flags)
        returns += reward
        reward=reward.reshape(10,1)
        reward_to_go-=reward
        if timestep >= max_timestep:
            break
        

    return np.mean(returns), np.std(returns)


def train(cfg, seed, log_dict, idx, logger, barrier, cmd):
    env_name_list = cfg.train.env_name
    transformer_parameter={}
    total_client = cfg.train.client_num[0] + cfg.train.client_num[1] + cfg.train.client_num[2]
    using_mp = barrier is not None

    if using_mp:
        local_log_dict = {key: [] for key in log_dict.keys()}
    else:
        local_log_dict = log_dict
        for key in local_log_dict.keys():
            for env_index in range(len(env_name_list)):
                local_log_dict[key][env_index].append([])
    utils.write_to_dict(local_log_dict, 'rtg_target', cfg.train.rtg_target, using_mp)
    for round in range(1,1+cfg.train.rounds):
        client_loss_array=np.array([])
        server_loss_array=np.array([])
        for env_index,env_name in enumerate(env_name_list):
            eval_env = gym.vector.make(env_name + '-v4', render_mode="rgb_array", num_envs=cfg.train.eval_episodes,
                                       asynchronous=False, wrappers=RecordEpisodeStatistics)
            utils.set_seed_everywhere(eval_env, seed)

            state_dim = utils.get_space_shape(eval_env.observation_space, is_vector_env=True)
            action_dim = utils.get_space_shape(eval_env.action_space, is_vector_env=True)
            buffer_server = instantiate(cfg.server, env_name=env_name,root_dir=cmd, seed=seed, )
            global_model = instantiate(cfg.model, state_dim=state_dim, action_dim=action_dim,
                                       action_space=eval_env.envs[0].action_space, state_mean=buffer_server.state_mean,
                                       state_std=buffer_server.state_std, device=device)


            if round!=1:
                global_model.load(f"{env_name}_global_model")
                new_state_dict = global_model.state_dict()
                for key in transformer_parameter:
                    new_state_dict[key] = transformer_parameter[key]
                global_model.load_state_dict(new_state_dict)

            eval_mean, eval_std = eval(eval_env, global_model, cfg.train.rtg_target[env_index])
            logger.info(
               f"Seed: {seed}, round: {round}, Env: {env_name}, Eval mean: {eval_mean:.2f}, Eval std: {eval_std:.2f}")
            env_sum_loss=train_global_client(env_name,env_index,eval_env,global_model,
                                             using_mp,local_log_dict,cfg,cmd)

            client_loss_array=np.append(client_loss_array,env_sum_loss)

        client_loss=client_loss_array.sum()/total_client
        utils.write_to_dict(local_log_dict, 'client_loss', client_loss, using_mp)



        for env_index,env_name in enumerate(env_name_list):
            eval_env = gym.vector.make(env_name + '-v4', render_mode="rgb_array", num_envs=cfg.train.eval_episodes,
                                       asynchronous=False, wrappers=RecordEpisodeStatistics)
            utils.set_seed_everywhere(eval_env, seed)
            state_dim = utils.get_space_shape(eval_env.observation_space, is_vector_env=True)
            action_dim = utils.get_space_shape(eval_env.action_space, is_vector_env=True)
            buffer = ClientBuffer(env_name=env_name, dataset=cfg.server.dataset, sample_type="traj_length",
                                  pos_encoding="absolute", context_len=cfg.model.context_len, gamma=1.0, root_dir=cmd,
                                  iid=True, client_index=0)
            model = DecisionTransformer(n_heads=1, n_blocks=3, drop_p=0.1, hidden_dim=128, context_len=cfg.model.context_len,
                                        reward_scale=1000, max_timestep=1000, state_dim=state_dim,
                                        action_dim=action_dim, action_space=eval_env.envs[0].action_space,
                                        state_mean=buffer.state_mean, state_std=buffer.state_std, device=device)

            model.load(f"{env_name}_global_model")

            transformer_parameter,env_sum_loss=train_transformer(env_name,env_index,cfg,model,transformer_parameter,local_log_dict,using_mp,cmd)
            server_loss_array=np.append(server_loss_array,env_sum_loss)
            if round % cfg.train.interval == 0:

                eval_mean, eval_std = eval(eval_env, model, cfg.train.rtg_target[env_index])
                utils.write_to_dict(local_log_dict, 'rounds', round, using_mp,env_index)
                utils.write_to_dict(local_log_dict, 'eval_returns', eval_mean, using_mp,env_index)
                d4rl_score = utils.get_d4rl_normalized_score(env_name, eval_mean)
                utils.write_to_dict(local_log_dict, 'd4rl_score', d4rl_score, using_mp,env_index)
                logger.info(f"Seed: {seed}, round: {round}, Env: {env_name}, Eval mean: {eval_mean:.2f}, Eval std: {eval_std:.2f}")
                if round % cfg.train.interval == 0:
                    utils.sync_and_visualize(log_dict, local_log_dict, barrier, idx, round,env_index,
                                             f'{env_name} ', using_mp)

        server_loss=server_loss_array.sum()/total_client
        utils.write_to_dict(local_log_dict, 'server_loss', server_loss, using_mp)
        utils.visualize_loss(round,"loss",local_log_dict)
    logger.info(f"Finish training seed {seed} with everage eval mean: {eval_mean}")


def checkpoint_train(cfg, seed, log_dict, idx, logger, barrier, cwd):
    env_name_list = cfg.train.env_name
    transformer_parameter={}
    total_client = cfg.train.client_num[0] + cfg.train.client_num[1] + cfg.train.client_num[2]
    using_mp = barrier is not None
    local_log_dict = log_dict
    for key in local_log_dict.keys():
        for env_index in range(len(env_name_list)):
            local_log_dict[key][env_index].append([])
    utils.write_to_dict(local_log_dict, 'rtg_target', cfg.train.rtg_target, using_mp)

    for round in range(cfg.checkpoint.cp_round,1+cfg.train.rounds):
        client_loss_array=np.array([])
        server_loss_array=np.array([])
        for env_index,env_name in enumerate(env_name_list):
            eval_env = gym.vector.make(env_name + '-v4', render_mode="rgb_array", num_envs=cfg.train.eval_episodes,
                                       asynchronous=False, wrappers=RecordEpisodeStatistics)
            utils.set_seed_everywhere(eval_env, seed)

            state_dim = utils.get_space_shape(eval_env.observation_space, is_vector_env=True)
            action_dim = utils.get_space_shape(eval_env.action_space, is_vector_env=True)
            buffer_server = instantiate(cfg.server, env_name=env_name,root_dir=cwd, seed=seed, )
            global_model = instantiate(cfg.model, state_dim=state_dim, action_dim=action_dim,
                                       action_space=eval_env.envs[0].action_space, state_mean=buffer_server.state_mean,
                                       state_std=buffer_server.state_std, device=device)

            if round==cfg.checkpoint.cp_round:
                dic=torch.load(cwd+cfg.checkpoint.dir+"/models/"+f"{env_name}_global_model.pt")
                global_model.load_state_dict(dic,strict=True)

            else:
                global_model.load(f"{env_name}_global_model")
                new_state_dict = global_model.state_dict().copy()
                for key in transformer_parameter:
                    new_state_dict[key] = transformer_parameter[key]
                global_model.load_state_dict(new_state_dict)

            eval_mean, eval_std = eval(eval_env, global_model, cfg.train.rtg_target[env_index])
            logger.info(f"Seed: {seed}, round: {round}, Env: {env_name}, Eval mean: {eval_mean:.2f}, Eval std: {eval_std:.2f}")
            env_sum_loss=train_global_client(env_name,env_index,eval_env,global_model,
                                             using_mp,local_log_dict,cfg,cwd)
            client_loss_array=np.append(client_loss_array,env_sum_loss)

        client_loss=client_loss_array.sum()/total_client
        utils.write_to_dict(local_log_dict, 'client_loss', client_loss, using_mp)



        for env_index,env_name in enumerate(env_name_list):
            eval_env = gym.vector.make(env_name + '-v4', render_mode="rgb_array", num_envs=cfg.train.eval_episodes,
                                       asynchronous=False, wrappers=RecordEpisodeStatistics)
            utils.set_seed_everywhere(eval_env, seed)
            state_dim = utils.get_space_shape(eval_env.observation_space, is_vector_env=True)
            action_dim = utils.get_space_shape(eval_env.action_space, is_vector_env=True)
            buffer = ClientBuffer(env_name=env_name, dataset=cfg.server.dataset, sample_type="traj_length",
                                  pos_encoding="absolute", context_len=cfg.model.context_len, gamma=1.0, root_dir=cwd,
                                  iid=True, client_index=0)
            model = DecisionTransformer(n_heads=1, n_blocks=3, drop_p=0.1, hidden_dim=128, context_len=cfg.model.context_len,
                                        reward_scale=1000, max_timestep=1000, state_dim=state_dim,
                                        action_dim=action_dim, action_space=eval_env.envs[0].action_space,
                                        state_mean=buffer.state_mean, state_std=buffer.state_std, device=device)

            model.load(f"{env_name}_global_model")
            transformer_parameter,env_sum_loss=train_transformer(env_name,env_index,cfg,model,transformer_parameter,local_log_dict,using_mp,cwd)
            server_loss_array=np.append(server_loss_array,env_sum_loss)
            if round % cfg.train.interval == 0:

                eval_mean, eval_std = eval(eval_env, model, cfg.train.rtg_target[env_index])
                utils.write_to_dict(local_log_dict, 'rounds', round, using_mp,env_index)
                utils.write_to_dict(local_log_dict, 'eval_returns', eval_mean, using_mp,env_index)
                d4rl_score = utils.get_d4rl_normalized_score(env_name, eval_mean)
                utils.write_to_dict(local_log_dict, 'd4rl_score', d4rl_score, using_mp,env_index)
                logger.info(f"Seed: {seed}, round: {round}, Env: {env_name}, Eval mean: {eval_mean:.2f}, Eval std: {eval_std:.2f}")
                if round % cfg.train.interval == 0:
                    utils.sync_and_visualize(log_dict, local_log_dict, barrier, idx, round,env_index,
                                             f'{env_name} ', using_mp)

        server_loss=server_loss_array.sum()/total_client
        utils.write_to_dict(local_log_dict, 'server_loss', server_loss, using_mp)
        utils.visualize_loss(round,"loss",local_log_dict)
    logger.info(f"Finish training seed {seed} with everage eval mean: {eval_mean}")
