import utils
import torch
import numpy as np
from model import DecisionTransformer
import torch.nn.functional as F
from buffer import ClientBuffer,ServerBuffer



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_global_client(env_name,env_index,eval_env,global_model,using_mp,local_log_dict,cfg,cmd):

    global_parameters = global_model.state_dict()
    state_dim = utils.get_space_shape(eval_env.observation_space, is_vector_env=True)
    action_dim = utils.get_space_shape(eval_env.action_space, is_vector_env=True)
    sum_parameters = None
    global_loss_array = np.array([])
    for client_index in range(cfg.train.client_num[env_index]):
        buffer = ClientBuffer(env_name=env_name, dataset=cfg.server.dataset, sample_type="traj_length",
                              pos_encoding="absolute", context_len=cfg.model.context_len, gamma=1.0, root_dir=cmd,
                              iid=True, client_index=client_index)

        model = DecisionTransformer(n_heads=1, n_blocks=3, drop_p=0.1, hidden_dim=128, context_len=cfg.model.context_len,
                                    reward_scale=1000, max_timestep=1000, state_dim=state_dim,
                                    action_dim=action_dim, action_space=eval_env.envs[0].action_space,
                                    state_mean=buffer.state_mean, state_std=buffer.state_std, device=device)
        model.load_state_dict(global_parameters, strict=True)
        #sequence_value = rewards_to_go
        local_parameters,avg_loss=train_certain_task(model,cfg,using_mp,local_log_dict,buffer,env_index)
        global_loss_array=np.append(global_loss_array,avg_loss)
        sum_parameters=utils.generate_sum_parameters(sum_parameters,local_parameters,global_parameters)


    client_sum_loss = global_loss_array.sum()
    #utils.write_to_dict(local_log_dict, 'client_loss', avg_loss.item(), using_mp)
    for key in global_parameters:
        if "block" not in key:
            global_parameters[key] = (sum_parameters[key] / cfg.train.client_num[env_index])

    global_model.load_state_dict(global_parameters, strict=True)
    #utils.unfreeze_model(global_model, "block")

    global_model.save(f"{env_name}_global_model")
    return client_sum_loss


def train_certain_task(model,cfg,using_mp,local_log_dict,buffer,env_index):
    loss_array = np.array([])
    utils.freeze_model(model, to_freeze="blocks")
    # cfg =cfg.train #DotMap(OmegaConf.to_container(cfg.train, resolve=True))
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda step: min((step + 1) / cfg.train.warmup_steps, 1))

    # logger.info(f"Training seed {seed} for {cfg.timesteps} timesteps with {model} and {buffer}")



    for timestep in range(1, cfg.train.client_timestep[env_index] + 1):
        states, actions, rewards_to_go, timesteps, mask = buffer.sample(cfg.train.batch_size)
        # no need for attention mask for the model as we always pad on the right side, whose attention is ignored by the casual mask anyway
        state_preds, action_preds, return_preds = model.forward(states, actions, rewards_to_go, timesteps)

        action_preds = action_preds[mask]
        action_loss = F.mse_loss(action_preds, actions[mask].detach(), reduction='mean')

        optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        scheduler.step()
        loss_array = np.append(loss_array, action_loss.cpu().detach().numpy())
    avg_loss = loss_array.mean()
    #utils.write_to_dict(local_log_dict, 'action_loss', avg_loss.item(), using_mp)
    utils.unfreeze_model(model, to_freeze="blocks")
    #model.save(f'model_client{client_index}')
    local_parameters = model.state_dict()
    return local_parameters,avg_loss



def train_transformer(env_name,env_index,cfg,model,transformer_parameter,local_log_dict,using_mp,cmd):
    '''for (name, param) in model.named_parameters():
        print(param.requires_grad)'''

    env_sum_loss=0
    for client_index in range(cfg.train.client_num[env_index]):
        buffer = ClientBuffer(env_name=env_name, dataset=cfg.server.dataset, sample_type="traj_length",
                              pos_encoding="absolute", context_len=cfg.model.context_len, gamma=1.0, root_dir=cmd,
                              iid=True, client_index=client_index)

        dic1=model.state_dict()
        for key in transformer_parameter:
            dic1[key] = transformer_parameter[key]
        model.load_state_dict(dic1)
        utils.freeze_model(model, "predict")
        utils.freeze_model(model, "embed")

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.lr,
                                      weight_decay=cfg.train.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lambda step: min((step + 1) / cfg.train.warmup_steps, 1))

        # logger.info(f"Training seed {seed} for {cfg.timesteps} timesteps with {model} and {buffer}")

        loss_array = np.array([])
        for timestep in range(1, cfg.train.server_timestep[env_index] + 1):
            states, actions, rewards_to_go, timesteps, mask = buffer.sample(cfg.train.batch_size)
            # no need for attention mask for the model as we always pad on the right side, whose attention is ignored by the casual mask anyway
            state_preds, action_preds, return_preds = model.forward(states, actions, rewards_to_go, timesteps)
            action_preds = action_preds[mask]
            action_loss = F.mse_loss(action_preds, actions[mask].detach(), reduction='mean')
            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()
            loss_array = np.append(loss_array, action_loss.cpu().detach().numpy())
        utils.unfreeze_model(model, "predict")
        utils.unfreeze_model(model, "embed")
        env_sum_loss+=loss_array.sum()
        for key, var in model.state_dict().items():
            #print(key)
            if "block" in key:
                transformer_parameter[key] = var
        #model.save(f"{env_name}_global_model")
    return transformer_parameter,env_sum_loss


