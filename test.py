import sys # for stderr, stdout
import logging
from argparse import ArgumentParser
import gymnasium as gym
import tandemtales_env.envs.tandem_gym as tandem_gym

class RandomAgent:
    def __init__(self):
        self.env = None
        self.action_mask = None
    def link(self, env):
        self.env = env
    def observe(self, observation, reward=0, terminated=False, truncated=False, info=None):
        if not info is None:
            self.action_mask = info.get('action_mask')
    def choose(self):
        if self.action_mask is None:
            return self.env.action_space.sample()
        else:
            return self.env.action_space.sample(self.action_mask)

def do_episode(env, agent, show=False, seed=None):
    agent.link(env)
    observation, info = env.reset(seed=seed)
    agent.observe(observation, info=info)
    episode_over = False
    total_reward = 0
    while not episode_over:
        action = agent.choose()
        observation, reward, terminated, truncated, info = env.step(action)
        agent.observe(observation, reward, terminated, truncated, info)
        if show: #info['learner_view'].display_turns('    ')
            print(env.render())
        total_reward += reward
        episode_over = terminated or truncated
    if show: print(f"Episode finished! Total reward: {total_reward}\n")
    return total_reward

if __name__ == '__main__':
    def main(args):
        rewarder = tandem_gym.RewardHandler()
        rewarder += tandem_gym.EndingScorer({'becameMonarch(Adventurer)': 100, 'playerDied()': -100, 'softlocked()': -100}, 50)
        rewarder += tandem_gym.RewardPass(GM=5, player=-10)
        rewarder += tandem_gym.RewardReplyYes(GM=5, player=5)
        rewarder += tandem_gym.RewardOfferYes(GM=-10)
        rewarder += tandem_gym.RewardAllNo(GM=-10, player=-10)
        rewarder += tandem_gym.RewardInvalid(-1000)
        print(f'Environment is in {"PLAYER" if args.player else "GAME MASTER"} mode.')
        player_first = not args.player if args.second else args.player
        print(f'PLAYER goes {"first" if player_first else "second"}.')
        env = gym.make('tandemtales_env/TandemTalesEnv-v0', world_model=args.model_file, render_mode='ansi',
            as_player=args.player, player_first=player_first, reward_handler=rewarder)#(action_rewards, ending_rewards))
        totals = []
        for i in range(2000):
            r = do_episode(env, RandomAgent(), show=False, seed=i)
            totals.append(r)
        for i in range(5):
            r = do_episode(env, RandomAgent(), show=True)
        print('average reward:', sum(totals)/len(totals))
        env.close()
        return 0

    parser = ArgumentParser(prog='TestTandemGym',
                        description='A program which tests the Tandem Tales Gym environment.')
    parser.add_argument('model_file', help='A JSON file describing the story model that should be used.')
    parser.add_argument('--player', action='store_true', help='A flag indicating that the single-agent environment should use the PLAYER role.')
    parser.add_argument('--second', action='store_true', help='A flag indicating that the chosen role should start second.')
    parser.add_argument('--ending', default='any', help='The reward scheme to use for endings')
    parser.add_argument('-l', '--loglevel', metavar='LEVEL', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='WARNING',
        help='The logging verbosity level, selected from DEBUG, INFO, WARNING, ERROR, or CRITICAL.', type=str.upper)
    parser.add_argument('-L', '--log', default='stderr', help='The name of a file or stream to log to.', type=str.lower)
    parser.add_argument('--profile', action='store_true', help='A flag indicating that main should be run with profiling turned on.')

    args = parser.parse_args()

    if args.log in {'stderr', '-', 'stdout'}:
        kwargs = {'stream': sys.stderr if args.log == 'stderr' else sys.stdout}
    else:
        kwargs = {'filename': args.log}
    logging.basicConfig(
        format='[%(asctime)s] (%(levelname)s|%(name)s|%(funcName)s) %(message)s', datefmt='%H:%M:%S',
        level=getattr(logging, args.loglevel.upper()), **kwargs)
    if args.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
    logging.info('Program started.')
    result = main(args)
    logging.info('Program completed.')
    if args.profile:
        profiler.disable()
        import pstats, io
        from pstats import SortKey
        stream = io.StringIO()
        statistics = pstats.Stats(profiler, stream=stream).sort_stats(SortKey.CUMULATIVE)
        statistics.print_stats(10)
        statistics = pstats.Stats(profiler, stream=stream).sort_stats(SortKey.TIME)
        statistics.print_stats(10)
        print(stream.getvalue())
        statistics.dump_stats('profile.out')

    exit(result)
