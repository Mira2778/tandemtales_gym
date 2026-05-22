from dataclasses import dataclass
from operator import attrgetter
import numpy as np
from tandemtales_env.world_logic import TT_ending

@dataclass
class TT_turn:
    role : str
    kind : str
    action : int|None
    replying : bool = False
    def __str__(self):
        return f'TT_turn({self.role}, {self.kind}, {self.action})'
    def get_context(self):
        result = {'ROLE': self.role, 'TURN_TYPE': self.kind}
        if not self.action is None:
            result.update(self.action.get_context())
        return result
    def is_pass(self):
        return self.kind == 'PASS'
    def is_positive(self):
        return self.kind in {'SUCCEED', 'FAIL'}
    def is_negative(self):
        return self.kind == 'FAIL'
    def by_system(self):
        return self.role == 'GAME_MASTER'
    def by_player(self):
        return self.role == 'PLAYER'

class GameView:
    def __init__(self, game):
        self.game = game
        self.role = None
        self.reset()
    def reset(self):
        self.described_index = 0
    def get_seen(self):
        state = self.game.story.world_states[-1]
        return self.game._world.get_encoded(state)
    def can_choose(self):
        return self.game.running() and self.game.active_role == self.role
    def get_turn_objects(self):
        turn_codes = self.get_turn_codes()
        return list(map(self.game.get_choice_from_id, turn_codes))
    def get_choice(self, choice_id):
        return self.game.get_choice_from_id(choice_id)
    def is_replying(self):
        return self.can_choose() and self.game.mode == 'REPLY'
    def describe_turns(self):
        undescribed_turns = []
        while self.described_index < len(self.game.turns):
            turn_entry = self.game.turns[self.described_index]
            described = self.game._world.describer.get_sentence(turn_entry, self.role)
            undescribed_turns.append(described)
            self.described_index += 1
        return undescribed_turns
    def display_turns(self, indent=''):
        for described in self.describe_turns():
            print(f'{indent}{described}')

class SystemView(GameView):
    def __init__(self, game):
        super().__init__(game)
        self.role = self.game._SYSTEM
    def get_turn_codes(self):
        if not self.can_choose(): return np.empty(0, dtype=np.uint)
        if not self.game.offer is None:
            return np.arange(self.game.offer.index, self.game.PASS_ID, self.game.N_ACTIONS, dtype=np.uint)
        choice_ids = []
        for action in self.game._world.get_tt_actions_in(self.game.story.world_states[-1]):
            if action.system_action:
                choice_ids.append(action.index)
                if not action.player_action:
                    choice_ids.append(action.index + self.game.N_ACTIONS)
        choice_ids.append(self.game.PASS_ID)
        return np.array(choice_ids, dtype=np.uint)
    def get_turn_codes_B(self):
        if not self.can_choose(): return np.empty(0, dtype=np.uint)
        if not self.game.offer is None:
            return np.arange(self.game.offer.index, self.game.PASS_ID, self.game.N_ACTIONS, dtype=np.uint)
        action_choice_mask = np.logical_and(self.game.world_action_mask, self.game.system_action_mask)
        propose_ids = self.game.world_actions[action_choice_mask]
        action_fail_mask = np.logical_and(action_choice_mask, np.logical_not(self.game.player_action_mask))
        fail_ids = self.game.world_actions[action_fail_mask] + self.game.N_ACTIONS
        return np.concat((propose_ids, fail_ids, np.array([self.game.PASS_ID], dtype=np.uint)))

class PlayerView(GameView):
    def __init__(self, game):
        super().__init__(game)
        self.role = self.game._PLAYER
    def get_turn_codes(self):
        if not self.can_choose(): return np.empty(0, dtype=np.uint)
        if not self.game.offer is None:
            return np.arange(self.game.offer.index, self.game.PASS_ID, self.game.N_ACTIONS, dtype=np.uint)
        choice_ids = []
        for action in self.game._world.get_tt_actions_in(self.game.story.world_states[-1]):
            if action.player_action:
                choice_ids.append(action.index)
        choice_ids.append(self.game.PASS_ID)
        return np.array(choice_ids, dtype=np.uint)
    def get_turn_codes_B(self):
        if not self.can_choose(): return np.empty(0, dtype=np.uint)
        if not self.game.offer is None:
            return np.arange(self.game.offer.index, self.game.PASS_ID, self.game.N_ACTIONS, dtype=np.uint)
        action_choice_mask = np.logical_and(self.game.world_action_mask, self.game.player_action_mask)
        propose_ids = self.game.world_actions[action_choice_mask]
        return np.concat((propose_ids, np.array([self.game.PASS_ID], dtype=np.uint)))

class GameModel:
    def __init__(self, world):
        self._world = world
        self._SYSTEM = 'GAME_MASTER'
        self._PLAYER = 'PLAYER'
        self.N_ACTIONS = len(self._world._actions)
        self.PASS_ID = 2*self.N_ACTIONS
        self.system_view = SystemView(self)
        self.player_view = PlayerView(self)
        self.views = {self._SYSTEM: self.system_view, self._PLAYER: self.player_view}
        # self.system_actions = np.array([action.index for action in self._world._actions if action.system_action], dtype=np.uint)
        # self.system_action_mask = np.zeros(self.N_ACTIONS, dtype=np.bool)
        # self.system_action_mask[self.system_actions] = True
        # self.player_actions = np.array([action.index for action in self._world._actions if action.player_action], dtype=np.uint)
        # self.player_action_mask = np.zeros(self.N_ACTIONS, dtype=np.bool)
        # self.player_action_mask[self.player_actions] = True
        # self.world_actions = np.arange(self.N_ACTIONS, dtype=np.uint)
        # self.world_action_mask = np.zeros(self.N_ACTIONS, dtype=np.bool)
        self.reset()
    def reset(self, player_first=False):
        self.story = self._world.get_empty_story()
        # self.update_world_action_mask()
        self.turns = []
        if player_first:
            self.commit_turn(self.PASS_ID)
        self.active_role = self._SYSTEM
        # self.choice_ids = []
        # self.update_choices()
        for view in self.views.values(): view.reset()
    def update_world_action_mask(self):
        valid = self._world.get_tt_actions_in(self.story.world_states[-1])
        self.world_action_mask.fill(False)
        # self.world_action_mask[self.world_action_mask] = False
        self.world_action_mask[list(map(attrgetter('index'), valid))] = True
    def running(self):
        return not self.story.terminated()
    @property
    def ending(self):
        if self.story.events and isinstance(self.story.events[-1], TT_ending):
            return self.story.events[-1]
    @property
    def mode(self):
        if self.turns and self.turns[-1].kind == 'PROPOSE':
            return 'REPLY'
        else:
            return 'OFFER'
    @property
    def offer(self):
        if self.mode == 'REPLY':
            return self.turns[-1].action
    def get_choice_from_id(self, choice_id, role=None, replying=None):
        if role is None: role = self.active_role
        if replying is None: replying = self.mode == 'REPLY'
        if choice_id == self.PASS_ID: return TT_turn(role, 'PASS', None)
        action_index = choice_id - (0 if choice_id < self.N_ACTIONS else self.N_ACTIONS)
        action = self._world.get_tt_action(action_index)
        if choice_id >= self.N_ACTIONS:
            kind = 'FAIL'
        elif replying or role == self._SYSTEM and not action.player_action:
            kind = 'SUCCEED'
        else:
            kind = 'PROPOSE'
        return TT_turn(role, kind, action, replying)
    def update_choices(self):
        # 1 + 2*len(self._world._actions) possible choices
        # 0 <= indexes < len(self._world._actions) are PROPOSE/SUCCEED turns (as appropriate) for the associated action
        # len(self._world._actions) <= indexes < 2*len(self.world._actions) are FAIL turns for the action associated with the index - len(self.world._actions)
        # index = len(self._world._actions) is PASS turn
        self.choice_ids.clear()
        if self.mode == 'REPLY':
            self.choice_ids.extend([self.offer.index, self.offer.index + len(self._world._actions)])
        elif self.active_role == self._SYSTEM:
            for action in self._world.get_tt_actions_in(self.story.world_states[-1]):
                if action.system_action:
                    tkind = 'PROPOSE' if action.player_action else 'SUCCEED'
                    self.choice_ids.append(action.index)
                    if not action.player_action:
                        self.choice_ids.append(action.index + len(self._world._actions))
            self.choice_ids.append(2*len(self._world._actions))
        elif self.active_role == self._PLAYER:
            for action in self._world.get_tt_actions_in(self.story.world_states[-1]):
                if action.player_action:
                    self.choice_ids.append(action.index)
            self.choice_ids.append(2*len(self._world._actions))
        else:
            raise NotImplementedError(f'{self.active_role}')
    def commit_turn(self, turn):
        if not self.running(): raise RuntimeError(f'game over, cannot take turns')
        if isinstance(turn, int): turn = self.get_choice_from_id(turn)
        self.turns.append(turn)
        if turn.kind in {'PASS', 'PROPOSE'}:
            if self.active_role == self._SYSTEM:
                self.active_role = self._PLAYER
            else:
                self.active_role = self._SYSTEM
        else:
            self.active_role = self._SYSTEM
        if turn.kind == 'SUCCEED':
            self._world.update_story(self.story, turn.action)
            # self.update_world_action_mask()
        # self.update_choices()