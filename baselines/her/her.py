import os

import click
import numpy as np
import json
import copy
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds, tf_util
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.mca import MCA
from baselines.her.metric_diversification import MetricDiversifier
np.set_printoptions(precision=6)


def save(epoch, policy, evaluator, rank, best_success_rate, save_path, save_interval, k):
    if evaluator.active:
        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        model_path = os.path.join(save_path, 'epoch_' + str(epoch) + '.model')
        model_path = os.path.expanduser(model_path)
        if rank == 0 and success_rate > best_success_rate and save_path:
            best_success_rate = success_rate
            save_message = f'(new best, rate: {best_success_rate:.2f}, best path: {model_path}, cover size: {k})'
            policy.save(model_path, message=save_message)
        if rank == 0 and save_interval > 0 and epoch % save_interval == 0 and save_path:
            save_message = f'(periodic, current best:{best_success_rate:.2f}, cover size: {k})'
            policy.save(model_path, message=save_message)
    return best_success_rate


def log(epoch, evaluator, rollout_worker, policy, rank, module):
    if rollout_worker.active:
        logger.record_tabular('module', module)
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))
        if rank == 0:
            logger.dump_tabular()


def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]


def train(*, policy, rollout_worker, evaluator, n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_path, demo_file, mca, random_cover=False, trainable=True, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    logger.info("Training...")
    best_success_rate = -1

    if policy.bc_loss == 1: policy.init_demo_buffer(demo_file)  # initialize demo buffer if training with demonstrations
    n_mca_envs = mca.rollout_worker.venv.num_envs
    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers

    n_init_epochs = 500

    for epoch in range(n_epochs):

        if epoch >= n_init_epochs:

            if mca.state_model.current_size == 101:
                import sys
                sys.exit(0)

            if epoch % 10 == 0:
                mca.state_model.save(save_path, message=f"K{mca.state_model.current_size}")
                mca.evaluator.generate_rollouts(ex_init=mca.state_model.draw(n_mca_envs), record=True, random=random_cover)
                mca.state_model.append_new_point(mca.tmp_point)

        # train
        rollout_worker.clear_history()
        mca.rollout_worker.clear_history()
        for n1 in range(n_cycles):
            random = (n1 % 2) == 0
            episode = rollout_worker.generate_rollouts()
            # mca.store_ex_episode(episode)
            mca_episode = mca.rollout_worker.generate_rollouts(ex_init=mca.state_model.draw(n_mca_envs),
                                                               random=random_cover or random or not trainable)

            mca.load_episode(mca_episode)

            if not trainable:
                continue

            episode = mca.overload_sg(episode, mca_episode)
            mca_episode = mca.overload_ss(mca_episode)

            # if random:
            #     continue

            policy.store_episode(episode)
            mca.policy.store_episode(mca_episode)

            for n2 in range(n_batches):
                policy.train()
                mca.policy.train()
            policy.update_target_net()
            mca.policy.update_target_net()

        # test
        evaluator.clear_history()
        mca.evaluator.clear_history()
        for n3 in range(n_test_rollouts):
            record = n3 == 0 and epoch % policy_save_interval == 0
            evaluator.generate_rollouts(record=record)
            mca.evaluator.generate_rollouts(ex_init=mca.state_model.draw(n_mca_envs), record=record, random=random_cover)
            # if record:
            #     mca.state_model.save(save_path, message=None)

        # record logs
        log(epoch, evaluator, rollout_worker, policy, rank, "policy")
        log(epoch, mca.evaluator, mca.rollout_worker, mca.policy, rank, "explorer")

        # save the policy if it's better than the previous ones
        best_success_rate = save(epoch, policy, evaluator, rank, best_success_rate, save_path + '/models',
                                 policy_save_interval, k=None)
        mca.best_success_rate = save(epoch, mca.policy, mca.evaluator, rank, mca.best_success_rate,
                                     save_path + '/mca_models', policy_save_interval, k=mca.state_model.current_size)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    return policy


def learn(*, network, env, mca_env, total_timesteps,
    seed=None,
    eval_env=None,
    replay_strategy='future',
    policy_save_interval=50,
    clip_return=True,
    demo_file=None,
    override_params=None,
    load_path=None,
    log_path=None,
    # save_path=None,
    **kwargs
):

    override_params = override_params or {}
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
        num_cpu = MPI.COMM_WORLD.Get_size()

    # Seed everything.
    rank_seed = seed + 1000000 * rank if seed is not None else None
    set_global_seeds(rank_seed)

    # assert operation mode
    assert kwargs["mode"] in ["basic", "exploration_module", "maximum_span"]

    if kwargs["mode"] == "basic":
        kwargs["mca_state_model"] = None

    def prepare_agent(_env, eval_env, active, exploration='eps_greedy', action_l2=None, scope=None, ss=False, load_path=None):
        # Prepare params.
        _params = copy.deepcopy(config.DEFAULT_PARAMS)
        _kwargs = copy.deepcopy(kwargs)
        _override_params = copy.deepcopy(override_params)

        env_name = _env.spec.id
        _params['env_name'] = env_name
        _params['replay_strategy'] = replay_strategy
        _params['ss'] = ss
        if action_l2 is not None:
            _params['action_l2'] = action_l2
        if not active:
            _params["buffer_size"] = 1
        if env_name in config.DEFAULT_ENV_PARAMS:
            _params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
        _params.update(**_override_params)  # makes it possible to override any parameter
        with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
             json.dump(_params, f)
        _params = config.prepare_params(_params)
        _params['rollout_batch_size'] = _env.num_envs

        if demo_file is not None:
            _params['bc_loss'] = 1
        _params.update(_kwargs)

        config.log_params(_params, logger=logger)

        if num_cpu == 1:
            logger.warn()
            logger.warn('*** Warning ***')
            logger.warn(
                'You are running HER with just a single MPI worker. This will work, but the ' +
                'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
                'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
                'are looking to reproduce those results, be aware of this. Please also refer to ' +
                'https://github.com/openai/baselines/issues/314 for further details.')
            logger.warn('****************')
            logger.warn()

        dims, coord_dict = config.configure_dims(_params)
        _params['ddpg_params']['scope'] = scope
        policy, reward_fun = config.configure_ddpg(dims=dims, params=_params, active=active, clip_return=clip_return)
        if load_path is not None:
            tf_util.load_variables(load_path)
            print(f"Loaded model: {load_path}")

        rollout_params = {
            'exploit': False,
            'use_target_net': False,
            'use_demo_states': True,
            'compute_Q': False,
            'exploration': exploration
        }

        eval_params = {
            'exploit': True,
            'use_target_net': _params['test_with_polyak'],
            'use_demo_states': False,
            'compute_Q': True,
        }

        for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
            rollout_params[name] = _params[name]
            eval_params[name] = _params[name]

        eval_env = eval_env or _env

        rollout_worker = RolloutWorker(_env, policy, dims, logger, active, monitor=True, **rollout_params)
        evaluator = RolloutWorker(eval_env, policy, dims, logger, active, **eval_params)

        return policy, rollout_worker, evaluator, _params, coord_dict, reward_fun

    active = kwargs["mode"] in ["basic", "exploration_module"]
    policy, rollout_worker, evaluator, params, *_ = prepare_agent(env, eval_env, active=active, scope="main")

    n_cycles = params['n_cycles']
    ##############################################################################
    # Maximum Coverage Agent
    mca_active = kwargs["mode"] in ["exploration_module", "maximum_span"]
    if 'mca_load_path' in kwargs:
        mca_load_path = kwargs['mca_load_path']
    else:
        mca_load_path = None

    mca_policy, mca_rw, mca_evaluator, mca_params, coord_dict, reward_fun = prepare_agent(mca_env, eval_env,
                                                                                          active=mca_active,
                                                                                          exploration=kwargs["mca_exploration"],
                                                                                          action_l2=kwargs["mca_action_l2"],
                                                                                          scope="mca",
                                                                                          ss=kwargs['ss'],  # True
                                                                                          load_path=mca_load_path
                                                                                          )

    load_p = 0.3
    phase_length = n_cycles * rollout_worker.T * mca_rw.rollout_batch_size * load_p

    mca_state_model = MetricDiversifier(k=kwargs['k'],
                                        reward_fun=reward_fun,
                                        vis=True,
                                        vis_coords=coord_dict['vis'],
                                        load_model=kwargs['load_mca_path'],
                                        save_path=log_path,
                                        random_cover=kwargs["random_cover"],
                                        load_p=load_p,
                                        phase_length=phase_length,
                                        dilute_at_goal=kwargs['dilute_at_goal'])

    mca = MCA(policy=mca_policy, rollout_worker=mca_rw, evaluator=mca_evaluator, state_model=mca_state_model,
              sharing=kwargs["sharing"], coord_dict=coord_dict, ss=kwargs['ss'])
    ##############################################################################

    if 'n_epochs' not in kwargs:
        n_epochs = total_timesteps // n_cycles // rollout_worker.T // mca_rw.rollout_batch_size
    else:
        n_epochs = int(kwargs['n_epochs'])

    # if kwargs["random_cover"]:
    #     random_interval = 1
    # else:
    #     random_interval = 2

    if 'trainable' in kwargs:
        trainable = kwargs['trainable']
    else:
        trainable = True

    return train(
        save_path=log_path, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, demo_file=demo_file, mca=mca, random_cover=kwargs['random_cover'],
        trainable=trainable)


@click.command()
@click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--total_timesteps', type=int, default=int(5e5), help='the number of timesteps to run')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--demo_file', type=str, default = 'PATH/TO/DEMO/DATA/FILE.npz', help='demo data file path')
@click.option('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)


def main(**kwargs):
    learn(**kwargs)


if __name__ == '__main__':
    main()
