import sys
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.run import parse_cmdline_kwargs
import gym


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    env = gym.make(args.env, **extra_args)
    env.reset()
    for i in range(100000):
        env.render()
        if i % 50 == 0:
            a = env.action_space.sample()
        obs, reward, done, info = env.step(a) # take a random action
        if not info['is_on_palm']:
            print("Not on palm")
            input()
    env.close()


if __name__ == '__main__':
    main(sys.argv)
