import hydra
import pickle
import os
import numpy as np

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def generate_dataset(cfg):
    for env_index,env_name in  enumerate(cfg.train.env_name):
        seed=cfg.seeds
        num_client=cfg.train.client_num[env_index]
        np.random.seed(seed)
        cwd=hydra.utils.get_original_cwd()
        dataset_path_expert = os.path.join(cwd, 'dataset', f'{env_name.lower()}-expert.pkl')
        dataset_path_medium = os.path.join(cwd, 'dataset', f'{env_name.lower()}-medium.pkl')
        dataset_path_replay = os.path.join(cwd, 'dataset', f'{env_name.lower()}-medium-replay.pkl')
        with open(dataset_path_expert, 'rb') as f:
            trajectories_expert = pickle.load(f)
        with open(dataset_path_medium, 'rb') as f:
            trajectories_medium = pickle.load(f)
        with open(dataset_path_replay, 'rb') as f:
            trajectories_replay = pickle.load(f)

        trajectories_e=[]
        trajectories_m = []
        trajectories_r = []
        trajectories_e.extend(trajectories_expert)
        trajectories_m.extend(trajectories_medium)
        trajectories_r.extend(trajectories_replay)

        num_trajs_e=len(trajectories_e)
        num_trajs_m = len(trajectories_m)
        num_trajs_r = len(trajectories_r)
        num_per_client_e=num_trajs_e//num_client
        num_per_client_m = num_trajs_m // num_client
        num_per_client_r = num_trajs_r // num_client

        partition_data_e={}
        partition_data_m = {}
        partition_data_r = {}
        if cfg.train.iid:
            for i in range(num_client):
                partition_data_e[i]=np.random.choice(trajectories_e,num_per_client_e,replace=False)
                partition_data_m[i] = np.random.choice(trajectories_m, num_per_client_m, replace=False)
                partition_data_r[i] = np.random.choice(trajectories_r, num_per_client_r, replace=False)
        else:
        # every client only concludes one type of data
            start_pointer=0
            for i in range(num_client):
                end_pointer=start_pointer+num_per_client_e
                partition_data_e[i]=trajectories_e[start_pointer:end_pointer]
                start_pointer=end_pointer
            start_pointer=0
            for i in range(num_client):
                end_pointer=start_pointer+num_per_client_m
                partition_data_m[i]=trajectories_m[start_pointer:end_pointer]
                start_pointer=end_pointer
            start_pointer=0
            for i in range(num_client):
                end_pointer=start_pointer+num_per_client_r
                partition_data_r[i]=trajectories_r[start_pointer:end_pointer]
                start_pointer=end_pointer

        distribution="iid" if cfg.train.iid else "non-iid"
        output_partition_file_e = os.path.join(cwd, 'dataset', f'{env_name.lower()}-expert-{distribution}.pkl')
        output_partition_file_m = os.path.join(cwd, 'dataset', f'{env_name.lower()}-medium-{distribution}.pkl')
        output_partition_file_r = os.path.join(cwd, 'dataset', f'{env_name.lower()}-medium-replay-{distribution}.pkl')
        with open(output_partition_file_e, "wb") as file:
            pickle.dump(partition_data_e, file)
        with open(output_partition_file_m, "wb") as file:
            pickle.dump(partition_data_m, file)
        with open(output_partition_file_r, "wb") as file:
            pickle.dump(partition_data_r, file)

if __name__=="__main__":
    generate_dataset()