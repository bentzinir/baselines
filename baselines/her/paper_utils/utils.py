import copy
from baselines.her.experiment import config
from baselines.common import tf_util
import matplotlib.pyplot as plt
from baselines.her.metric_diversification import MetricDiversifier
from baselines.her.paper_utils.env_scripts import reset_env, init_from_point
import numpy as np
import os


def load_policy(env_id, **kwargs):
    params = config.DEFAULT_PARAMS
    _override_params = copy.deepcopy(kwargs)
    params.update(**_override_params)
    params['env_name'] = env_id
    params = config.prepare_params(params)
    dims, coord_dict = config.configure_dims(params)
    params['ddpg_params']['scope'] = "mca"
    policy, reward_fun = config.configure_ddpg(dims=dims, params=params, active=True, clip_return=True)
    return policy, reward_fun


def load_model(load_path):
    tf_util.load_variables(load_path)
    print(f"Loaded model: {load_path}")


def exp1_to_figure(results, save_directory, alpha, message=""):
    fig, ax = plt.subplots(1, 1)

    def cover_plot(data, name):
        y = data["mean"]
        x = data["time"]
        ax.plot(x, y, label=f"k={name}")
        if "std" in data:
            error = data["std"]
            ax.fill_between(x, y - error, y + error, alpha=0.5)

    for key, val in results.items():
        cover_plot(data=val, name=key)
    ax.legend(loc=0, prop={'size': 15})

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    fig_name = f"{save_directory}/{message}_packing.png"
    plt.title(r'$\alpha =$' + f"{alpha}", fontsize=20)
    plt.xlabel("Iters.", fontsize=20)
    plt.ylabel(r'$\epsilon_d(\mathcal{M})$', fontsize=20)
    plt.ylim([0, 10])
    plt.locator_params(nbins=4)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.savefig(fig_name)

    print(f"saved figure : {fig_name}")


def exp1_overlayed_figure(env, scrb:MetricDiversifier, save_directory, message):

    reset_env(env, scrb, mode='intrinsic')

    rooms_layer = env.env._get_rooms_image()
    agent_layer = env.env._get_agent_image()

    for pidx in scrb.used_slots():
        # obs = reset_env(env, scrb, mode='intrinsic')
        init_from_point(env, scrb.buffer[pidx])
        agent_p = env.env._get_agent_image()
        agent_layer += agent_p

    agent_layer = (255 * (agent_layer / agent_layer.max())).astype(np.int32)
    frame = np.concatenate([agent_layer, 0 * rooms_layer, rooms_layer], axis=2)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i, j, :].sum() == 0:
                frame[i, j] = 255

    fig, ax = plt.subplots(1, 1)
    plt.imshow(frame)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    fig_name = f"{save_directory}/{message}_frame.png"
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    plt.tight_layout()
    plt.savefig(fig_name)

    print(f"saved figure : {fig_name}")


def list_epochs(directory):
    model_names = os.listdir(directory)
    return [int(model_name.split('_')[1].split('.')[0]) for model_name in model_names]


def exp2_to_figure(results, save_directory):
    fig, ax = plt.subplots(1, 1)

    def cover_plot(data, name):
        y = data["mean"]
        x = data["epochs"]
        ax.plot(x, y, label=name)
        if "std" in data:
            error = data["std"]
            ax.fill_between(x, y - error, y + error, alpha=0.5)

    for key, val in results.items():
        cover_plot(data=val, name=results[key]["method_name"])
    ax.legend(loc=0, prop={'size': 15})

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    fig_name = f"{save_directory}/cover.png"
    plt.title("Goal Coverage", fontsize=20)
    plt.xlabel("Epochs", fontsize=20)
    # plt.ylabel(f"%", fontsize=20)
    plt.ylim([0, 1])
    plt.locator_params(nbins=4)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.savefig(fig_name)

    print(f"saved figure : {fig_name}")