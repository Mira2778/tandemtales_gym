from typing import Optional
import numpy as np
import gymnasium as gym # https://gymnasium.farama.org/
from tandemtales_env.world_logic import load_world_model
from tandemtales_env.game_logic import GameModel

class PartnerAgent:
    def choose(self, generator, view): raise NotImplementedError()

class RandomPartner(PartnerAgent):
    def __init__(self, do_pass, do_fail):
        self.do_pass = do_pass
        self.do_fail = do_fail
    def __str__(self):
        return f'RandomPartner: {self.do_pass:.1%} chance to offer PASS, {self.do_fail:.1%} chance to reply FAIL'
    def choose(self, generator, view):
        turn_choices = view.get_turn_objects()
        if view.is_replying():
            if generator.random() < self.do_fail:
                choice = turn_choices[-1] # should be the FAIL choice
                if choice.kind != 'FAIL': raise RuntimeError(f'{choice}')
            else:
                choice = turn_choices[0] # should be the SUCCEED choice
                if choice.kind != 'SUCCEED': raise RuntimeError(f'{choice}')
        elif generator.random() < self.do_pass or len(turn_choices) == 1:
            choice = turn_choices[-1] # should be the PASS option
            if choice.kind != 'PASS': raise RuntimeError(f'{choice}')
        else:
            choice = generator.choice(turn_choices[:-1])
        return choice

class RewardHandler:
    def __init__(self, children=None):
        self.reward = 0
        self.children = [] if children is None else children
    def __add__(self, other):
        return RewardHandler(self.children + [other])
    def __radd__(self, other):
        return RewardHandler(self.children + [other])
    def __iadd__(self, other):
        self.children.append(other)
        return self
    def on_reset(self, game, role):
        for child in self.children:
            child.on_reset(game, role)
    # occurs after partner steps, so has the option to clear reward
    #  (from before first observation) or otherwise react to initial
    #  conditions. role identifies who the reward is targeted towards
    #  for all calls until next reset (reward is already drained after)
    def on_turn(self, game, turn, partner=False): 
        for child in self.children:
            child.on_turn(game, turn, partner)
    # on_turn occurs whenever a step choice is submitted, including
    #  partner steps. it occurs just AFTER the choice is committed
    #  (or instead of a choice being committed, if allow_invalid=True)
    # will eventually make game some kind of view-object to make sure there
    #  aren't any side effects
    def on_ending(self, game, ending):
        for child in self.children:
            child.on_ending(game, ending)
    # if the game reaches an ending and terminates normally, on_ending gets
    #  called once just before the environment terminates, with the ending
    #  that was reached. (could also be None if the underlying story was
    #  somehow early-terminated) may have prexisting reward from 
    def drain(self):
        result = self.reward
        self.reward = 0
        for child in self.children:
            result += child.drain()
        return result

class EndingScorer(RewardHandler):
    def __init__(self, scores, default):
        super().__init__()
        self.scores = scores # use '' for None ending (not expected to happen?)
        self.default = default # None won't give default
    def on_ending(self, game, ending):
        super().on_ending(game, ending)
        if ending is None:
            if '' in self.scores:
                self.reward += self.scores['']
        else:
            self.reward += self.scores.get(ending.name, self.default)

class RewardPass(RewardHandler):
    def __init__(self, GM=None, player=None):
        super().__init__()
        self.GM = GM
        self.player = player
    def on_turn(self, game, turn, partner=False): 
        super().on_turn(game, turn, partner)
        if not turn is None and turn.is_pass():
            if not self.GM is None and turn.by_system():
                self.reward += self.GM
            if not self.player is None and turn.by_player():
                self.reward += self.player

class RewardReplyYes(RewardHandler):
    def __init__(self, GM=None, player=None):
        super().__init__()
        self.GM = GM
        self.player = player
    def on_turn(self, game, turn, partner=False): 
        super().on_turn(game, turn, partner)
        if not turn is None and turn.replying and turn.is_positive():
            if not self.GM is None and turn.by_system():
                self.reward += self.GM
            if not self.player is None and turn.by_player():
                self.reward += self.player

class RewardOfferYes(RewardHandler):
    def __init__(self, GM=None, player=None):
        super().__init__()
        self.GM = GM
        self.player = player
        if not player is None:
            raise TypeError('this type of event does not occur for the player')
    def on_turn(self, game, turn, partner=False): 
        super().on_turn(game, turn, partner)
        if not turn is None and not turn.replying and turn.kind == 'SUCCEED':
            if not self.GM is None and turn.by_system():
                self.reward += self.GM

class RewardAllNo(RewardHandler):
    def __init__(self, GM=None, player=None):
        super().__init__()
        self.GM = GM
        self.player = player
    def on_turn(self, game, turn, partner=False): 
        super().on_turn(game, turn, partner)
        if not turn is None and turn.is_negative():
            if not self.GM is None and turn.by_system():
                self.reward += self.GM
            if not self.player is None and turn.by_player():
                self.reward += self.player

class RewardInvalid(RewardHandler):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def on_turn(self, game, turn, partner=False): 
        super().on_turn(game, turn, partner)
        if turn is None:
            self.reward += self.value
# Want to do this better
# want a system for saying:
#   if a turn passes from the GM to the player, give this reward
#   if a turn is a succeeding action on the reply turn, apply this cost
#   if a turn is a fail action on the reply turn, apply this cost
#   if a turn is a succeeding/failing action on the offer turn (GM implied), apply this cost
#   if a turn is an offer action, apply this cost (i.e. cost per step)
#   if an invalid action was attempted, apply this cost
#   if the partner was able to satisfy a certain condition...
#   if this ending is achieved, give this reward/penalty (once)
#   if no ending is achieved, apply this penalty
#   if any of these endings is achieved, apply this reward/penalty
#   etc?
# ideally it gets stored in a file
# (Also, ideally this could be substituted with a critic?)

class TandemTalesEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 1}
    def __init__(self, render_mode=None, world_model=None, allow_invalid=False, as_player=False, player_first=False, reward_handler=None, partner_agent=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        if world_model is None: raise ValueError(f'a world_model is required')
        if isinstance(world_model, str):
            world_model = load_world_model(world_model)
        self.render_mode = render_mode
        self.world = world_model
        self.allow_invalid = allow_invalid
        self.game = GameModel(world_model)
        if as_player:
            self.learner_role = self.game._PLAYER
            self.partner_role = self.game._SYSTEM
        else:
            self.learner_role = self.game._SYSTEM
            self.partner_role = self.game._PLAYER
        self.learner_view = self.game.views[self.learner_role]
        self.partner_view = self.game.views[self.partner_role]
        self.player_first = player_first
        if partner_agent is None: partner_agent = RandomPartner(do_pass=0.3, do_fail=0.1)
        self.partner_agent = partner_agent
        if reward_handler is None: reward_handler = RewardHandler()
        self.rewarder = reward_handler
        self.observation_space = gym.spaces.MultiBinary(self.world.state_encoding_width + 2, seed=self.np_random)
        action_width = len(self.world._actions) * 2 + 1
        self.action_space = gym.spaces.Discrete(action_width, dtype=np.int32, seed=self.np_random)
        self._obs_vector = np.zeros(self.observation_space.shape[0], dtype=np.bool)
        self._action_mask = np.zeros(self.action_space.n, dtype=np.int8)
        self._action_mask_b = self._action_mask.view(np.bool)
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.rewarder.drain()
        self.game.reset(player_first=self.player_first)
        # if as_player=True and player_first=False, or as_player=False and player_first=True
        # the partner agent may act before the first observation
        self.do_partner_steps() 
        # the partner might end the game before it starts, so prevent this
        while not self.game.running():
            self.game.reset(player_first=self.player_first)
            self.do_partner_steps()
        self.rewarder.on_reset(self.game, self.learner_role)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    def _get_obs(self):
        turn_vector = np.array([0, 1] if self.game.mode == 'REPLY' else [1, 0])
        state_vector = self.learner_view.get_seen()
        self._obs_vector[:2] = turn_vector
        self._obs_vector[2:] = state_vector
        # turn_vector, offer_vector, state_vector, valid_actions = self.learner_view.turn_data()
        # self._obs_vector.fill(0)
        # offset = len(state_vector)
        # self._obs_vector[:offset] = np.array(turn_vector + offer_vector + state_vector, dtype=np.bool)
        # self._obs_vector[[offset + act.index for act in valid_actions]] = 1
        return self._obs_vector
    def _get_info(self):
        self._action_mask.fill(0)
        self._action_mask[self.learner_view.get_turn_codes()] = 1
        return {'action_mask': self._action_mask,
            'b_action_mask': self._action_mask_b,
            '_learner_view': self.learner_view,
            '_partner_view': self.partner_view
        }
    def action_masks(self): return self._get_info()['action_mask']
    def step(self, choice_id):
        choice_invalid = choice_id not in self.learner_view.get_turn_codes()
        if choice_invalid:
            if not self.allow_invalid:
                raise RuntimeError(f'{choice_id} is not a valid option for the current turn')
            choice = None
            self.rewarder.on_turn(self.game, None)
        else:
            choice = self.learner_view.get_choice(choice_id)
            self.game.commit_turn(choice)
            self.rewarder.on_turn(self.game, choice)
            self.do_partner_steps()
        observation = self._get_obs()
        info = self._get_info()
        truncated = False
        terminated = not self.game.running()
        if terminated:
            self.rewarder.on_ending(self.game, self.game.ending)
        if self.render_mode == 'human':
            self._render_frame()
        return observation, self.rewarder.drain(), terminated, truncated, info
    def render(self):
        if self.render_mode == 'ansi':
            return self._render_frame()
    def _render_frame(self):
        if self.render_mode == 'ansi':
            return " ".join(self.learner_view.describe_turns())
        elif self.render_mode == 'human':
            pass
    def do_partner_steps(self):
        while self.partner_view.can_choose():
            choice = self.partner_agent.choose(self.np_random, self.partner_view)
            self.game.commit_turn(choice)
            self.rewarder.on_turn(self.game, choice, True)