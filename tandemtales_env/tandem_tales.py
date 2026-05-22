import sys # for stderr, stdout
import logging
from argparse import ArgumentParser
from pathlib import Path
import json
from operator import methodcaller, attrgetter, itemgetter
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any
import numpy as np

class Proposition:
    def trivially_true(self):
        return False
    def trivially_false(self):
        return False

class Conjunction(Proposition):
    def __init__(self, args=None):
        self.args = () if args is None else args
    def __str__(self):
        if len(self.args) == 0:
            return 'True'
        else:
            return ' & '.join(map(str, self.args))
    def __eq__(self, other):
        return isinstance(other, Conjunction) and frozenset(self.args) == frozenset(other.args)
    def __hash__(self):
        return hash(self.args)
    def trivially_true(self):
        return len(self.args) == 0
    def evaluate_in(self, state):
        return all(map(methodcaller('evaluate_in', state), self.args))
    def get_variables(self):
        return set().union(*map(methodcaller('get_variables'), self.args))
    def get_equality_tests(self):
        return set().union(*map(methodcaller('get_equality_tests'), self.args))
    def discard_variables(self, to_remove):
        return Conjunction(tuple(map(methodcaller('discard_variables', to_remove), self.args)))
    def replace_tests(self, test_refs):
        return Conjunction(tuple(map(methodcaller('replace_tests', test_refs), self.args)))
    def test_in(self, s):
        return all(map(methodcaller('test_in', s), self.args))

class Disjunction(Proposition):
    def __init__(self, args=None):
        self.args = () if args is None else args
    def __str__(self):
        if len(self.args) == 0:
            return 'False'
        elif len(self.args) == 1:
            return str(self.args[0])
        else:
            return '(' + ' | '.join(map(str, self.args)) + ')'
    def __eq__(self, other):
        return isinstance(other, Disjunction) and frozenset(self.args) == frozenset(other.args)
    def __hash__(self):
        return hash(self.args)
    def trivially_false(self):
        return len(self.args) == 0
    def evaluate_in(self, state):
        return any(map(methodcaller('evaluate_in', state), self.args))
    def get_variables(self):
        return set().union(*map(methodcaller('get_variables'), self.args))
    def discard_variables(self, to_remove):
        return Disjunction(tuple(map(methodcaller('discard_variables', to_remove), self.args)))
    def get_equality_tests(self):
        return set().union(*map(methodcaller('get_equality_tests'), self.args))
    def replace_tests(self, test_refs):
        return Disjunction(tuple(map(methodcaller('replace_tests', test_refs), self.args)))
    def test_in(self, s):
        return any(map(methodcaller('test_in', s), self.args))
        
class Negation(Proposition):
    def __init__(self, arg):
        self.arg = arg
    def __str__(self):
        if isinstance(self.arg, (Disjunction, Conjunction)) and len(self.arg.args) > 1:
            return f'!({self.arg})'
        else:
            return f'!{self.arg}'
    def __eq__(self, other):
        return isinstance(other, Negation) and self.arg == other.arg
    def __hash__(self):
        return hash(('negate', self.arg))
    def evaluate_in(self, state):
        return not self.arg.evaluate_in(state)
    def get_variables(self):
        return self.arg.get_variables()
    def discard_variables(self, to_remove):
        return Negation(self.world, self.arg.discard_variables(to_remove))
    def get_equality_tests(self):
        return self.arg.get_equality_tests()
    def replace_tests(self, test_refs):
        return Negation(self.arg.replace_tests(test_refs))
    def test_in(self, s):
        return not self.arg.test_in(s)

@dataclass(frozen=True)
class TT_value:
    index : int
    value : str|bool|None
    name : str
    description : str
    def __str__(self):
        return self.name
    def __lt__(self, other):
        if not isinstance(other, TT_value): raise NotImplementedError(f'{self} < {other}')
        return self.index < other.index
    @property
    def is_constant(self):
        return self.index < 0
    def get_context(self):
        if self.is_constant:
            return {'TYPE': 'constant', 'CONSTANT': self, 'NAME': self.name}
        else:
            return {'TYPE': 'entity', 'ENTITY': self, 'NAME': self.name}

@dataclass
class TT_domain:
    index : int
    values : Sequence[TT_value]
    constants : Sequence[bool|None]
    encodings : Sequence[np.bool]
    variables : Sequence[int]
    boolean : bool = False
    def __str__(self):
        return f'domain for {", ".join(map(str, self.values))}'
    def get_fact(self, value):
        if value not in self.values:
            return None
        internal_id = self.values.index(value)
        flipped = None
        if self.boolean:
            flipped_id = (internal_id + 1)%2
            flipped = (self.values[flipped_id], flipped_id)
        return TT_fact(self.index, value, internal_id, flipped)

@dataclass(frozen=True)
class TT_fact:
    domain_id : int
    value : TT_value
    code_index : int
    flipped : Any = field(default=None, compare=False)
    def __str__(self):
        return self.value.name
    @property
    def boolean(self):
        return not self.flipped is None
    def negated(self):
        if self.flipped is None:
            raise RuntimeError('cannot directly negate fact of a non-boolean domain')
        return TT_fact(self.domain_id, *self.flipped, (self.value, self.code_index))

@dataclass(frozen=True)
class TT_signature:
    name : str
    args : Sequence[TT_value]
    def __str__(self):
        return f'{self.name}({", ".join(map(str, self.args))})'
    def get_context(self):
        result = {'NAME': self.name}
        result.update({f'ARG_{i}': arg for i, arg in enumerate(self.args)})
        return result

@dataclass(frozen=True)
class TT_variable:
    index : int
    name : str
    description : str
    domain_id : int
    initial : TT_value
    signature : TT_signature
    offset : int
    width : int
    def __str__(self):
        return self.name
    @property
    def state_idx(self):
        return list(range(self.offset, self.offset + self.width))
    def get_context(self):
        result = {'TYPE': 'variable', 'VARIABLE': self, 'NAME': self.name}
        result.update(self.signature.get_context())
        return result

@dataclass(frozen=True)
class TT_equality:
    lhs : TT_variable
    rhs : TT_variable|TT_fact
    def __str__(self):
        return ' == '.join(map(str, (self.lhs, self.rhs)))
    def test_in(self, state):
        l_fact = state.facts[self.lhs.index]
        r_fact = state.facts[self.rhs.index] if isinstance(self.rhs, TT_variable) else self.rhs
        if l_fact.domain_id != r_fact.domain_id:
            l_fact, r_fact = l_fact.value, r_fact.value
        return l_fact == r_fact
    def get_equality_tests(self):
        return {self}
    def replace_tests(self, test_refs):
        return test_refs.get(self, self)

@dataclass(frozen=True)
class TT_test_reference:
    index : int
    world : Any = field(default=None, compare=False)
    def __str__(self):
        return str(self.world._stored_tests[index])
    def test_in(self, state, tested=False):
        if tested:
            return state[self.index]
        else:
            return self.world._stored_tests[self.index].test_in(state)
    def get_equality_tests(self):
        return set()

@dataclass
class TT_assignment:
    variable : TT_variable
    fact : TT_fact
    def __str__(self):
        return f'{var} := {fact}'
    def get_context(self):
        result = {'TYPE': 'assignment', 'VALUE': self.fact.value}
        result.update(self.variable.signature.get_context())
        return result

@dataclass
class TT_assign:
    variables : Sequence[TT_variable]
    facts : Sequence[TT_fact]
    def __str__(self):
        return '; '.join(map(str, self.parts))
    @property
    def parts(self):
        return [TT_assignment(var, fact) for var, fact in zip(self.variables, self.facts)]

@dataclass
class TT_effect:
    unconditional : TT_assign
    conditional : Sequence[TT_assign]
    requirements : Sequence[TT_equality|Proposition]
    def __str__(self):
        parts = [str(self.unconditional)] if self.unconditional.variables else []
        for assign, condition in zip(self.conditional, self.requirements):
            if len(assign.variables) > 1:
                parts.append(f'({assign}) when {condition}')
            else:
                parts.append(f'{assign} when {condition}')
        return '; '.join(parts)
    def resolve_in(self, state):
        if self.conditional:
            variables = list(self.unconditional.variables)
            facts = list(self.unconditional.facts)
            for assign, req in zip(self.conditional, self.requirements):
                if req.test_in(state): # TEST_HERE
                    variables.extend(assign.variables)
                    facts.extend(assign.facts)
            return TT_assign(variables, facts)
        else:
            return self.unconditional
    def get_equality_tests(self):
        return set().union(*map(methodcaller('get_equality_tests'), self.requirements))
    def replace_tests(self, test_refs):
        return TT_effect(self.unconditional, self.conditional,
            tuple(map(methodcaller('replace_tests', test_refs), self.requirements)))

@dataclass
class TT_action:
    index : int
    name : str
    description : str
    signature : TT_signature
    precondition : Proposition
    effect : TT_effect
    visibility : Proposition
    consenting : Sequence[TT_value]
    player_action : bool
    system_action : bool
    def __str__(self):
        return self.name
    def __eq__(self, other):
        return isinstance(other, TT_action) and self.index == other.index
    def get_context(self):
        result = {'TYPE': 'action', 'ACTION': self, 'NAME': self.name}
        result.update(self.signature.get_context())
        return result
    def get_equality_tests(self):
        return set().union(*map(methodcaller('get_equality_tests'),
                    (self.precondition, self.effect, self.visibility)))
    def replace_tests(self, test_refs):
        precondition = self.precondition.replace_tests(test_refs)
        effect = self.effect.replace_tests(test_refs)
        visibility = self.visibility.replace_tests(test_refs)
        return TT_action(self.index, self.name, self.description, self.signature,
            precondition, effect, visibility, self.consenting,
            self.player_action, self.system_action)

@dataclass(frozen=True)
class TT_ending:
    index : int
    name : str
    description : str
    signature : TT_signature
    precondition : Proposition
    def __str__(self):
        return f'{self.name}: {self.description}'
    def get_context(self):
        result = {'TYPE': 'ending', 'ENDING': self, 'NAME': self.name}
        result.update(self.signature.get_context())
        return result
    def get_equality_tests(self):
        return self.precondition.get_equality_tests()
    def replace_tests(self, test_refs):
        precondition = self.precondition.replace_tests(test_refs)
        return TT_ending(self.index, self.name, self.description, self.signature, precondition)

@dataclass
class TT_state:
    facts : Sequence[TT_fact]
    def state_after(self, assigner):
        new_facts = list(self.facts)
        for var, fact in zip(assigner.variables, assigner.facts):
            new_facts[var.index] = fact
        return TT_state(new_facts)

@dataclass
class TT_cached_state:
    cache : Any
    index : int
    visibility : bool = False
    @property
    def raw_state(self):
        if self.visibility:
            return self.cache._vis_states[self.index]
        else:
            return self.cache._states[self.index]
    @property
    def facts(self):
        return self.raw_state.facts
    def state_after(self, assigner):
        if self.visibility:
            raise NotImplementedError()
        curr_state = self.cache._states[self.index]
        next_index = self.cache.insert_state(curr_state.state_after(assigner))
        return TT_cached_state(self.cache, next_index, self.visibility)

# Story is a sequence of world states which happened, a parallel sequence of visibility states
#  and a sequence of actions ending with TT_ending or None
@dataclass
class TT_story:
    world_states : Sequence[TT_state|TT_cached_state]
    visible_states : Sequence[TT_state|TT_cached_state]
    events : Sequence[TT_action|TT_ending|None] # None if it didn't end
    def terminated(self):
        if self.events:
            return self.events[-1] is None or isinstance(self.events[-1], TT_ending)
        else:
            return False

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

class GameView:
    def __init__(self, game):
        self.game = game
        self.role = None
        self.reset()
    def reset(self):
        self.display_index = 0
    def get_seen(self):
        state = self.game.story.world_states[-1]
        return self.game._world.get_encoded(state)
    def can_choose(self):
        return self.game.running() and self.game.active_role == self.role
    def get_choices(self, indexes=False):
        if self.can_choose():
            valid = self.game.choice_ids
            if indexes:
                return valid
            else:
                return list(map(self.game.get_choice_from_id, valid))
        else:
            return []
    def get_choice(self, choice_id):
        return self.game.get_choice_from_id(choice_id)
    def is_replying(self):
        return self.can_choose() and self.game.mode == 'REPLY'
    def display_turns(self, indent=''):
        while self.display_index < len(self.game.turns):
            turn_entry = self.game.turns[self.display_index]
            described = self.game._world.describer.get_sentence(turn_entry, self.role)
            print(f'{indent}{described}')
            self.display_index += 1

class SystemView(GameView):
    def __init__(self, game):
        super().__init__(game)
        self.role = self.game._SYSTEM

class PlayerView(GameView):
    def __init__(self, game):
        super().__init__(game)
        self.role = self.game._PLAYER

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
        self.reset()
    def reset(self):
        self.story = self._world.get_empty_story()
        self.turns = []
        self.active_role = self._SYSTEM
        self.choice_ids = []
        self.update_choices()
        for view in self.views.values(): view.reset()
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
        self.update_choices()

class WorldCache:
    def __init__(self, model):
        self._model = model
        self.ALLOC_BATCH = 1000000
        self._states = []
        self._vis_states = []
        self._valid_actions = []
        self._child_states = []
        self._endings = []
        self._encodings = np.zeros((self.ALLOC_BATCH,self._model.state_encoding_width), dtype=np.bool)
        self._vis_encodings = np.zeros((self.ALLOC_BATCH,self._model.state_encoding_width), dtype=np.bool)
        self._diffs = set()
        self._diffs_2 = set()
        self.insert_state(model._initial, initial=True)
    def __str__(self):
        duplicated = len(self._states) - len(self._diffs)
        duplicated2 = len(self._states) - len(self._diffs_2)
        return f'WorldCache({self._model.name}) with {len(self._states)} cached states ({len(self._diffs)} initdiffs {duplicated/len(self._states):.2%} duplicated; {len(self._diffs_2)} with vis {duplicated2/len(self._states):.2%} duplicated)'
    def get_init_diff(self, state):
        return frozenset(((var.index, state.facts[var.index].value) for var in self._model._variables if state.facts[var.index].value != var.initial))
    def get_state(self, index, wrap=True, vis=False):
        if wrap:
            return TT_cached_state(self, index, vis)
        elif vis:
            return self._vis_states[index]
        else:
            return self._states[index]
    def insert_state(self, state, initial=False):
        insert_index = len(self._states)
        if insert_index >= self._encodings.shape[0]:
            np.append(self._encodings, np.zeros((self.ALLOC_BATCH,self._model.state_encoding_width), dtype=np.bool), axis=0)
            np.append(self._vis_encodings, np.zeros((self.ALLOC_BATCH,self._model.state_encoding_width), dtype=np.bool), axis=0)
            print(self)
        valid = self._model.get_tt_actions_in(state, no_cache=True)
        self._states.append(state)
        self._diffs.add(self.get_init_diff(state))
        # self._encodings.append(self._model.get_encoded(state, no_cache=True))
        # offset = insert_index*self._model.state_encoding_width
        # end = offset + self._model.state_encoding_width
        self._encodings[insert_index,:] = self._model.get_encoded(state, no_cache=True)
        if initial:
            self._vis_states.append(state)
            self._vis_encodings[insert_index,:] = self._model.get_encoded(state, no_cache=True)
            self._diffs_2.add((self.get_init_diff(state), self.get_init_diff(state)))
            # self._vis_encodings.append(self._model.get_encoded(state, no_cache=True))
        else:
            self._vis_states.append(None)
            # self._vis_encodings.append(None)
        self._valid_actions.append(tuple(map(attrgetter('index'), valid)))
        self._child_states.append({})
        self._endings.append(self._model.get_tt_endings_in(state, no_cache=True))
        return insert_index
    def get_next_state(self, state, action, wrap=True):
        if isinstance(action, int): action = self._model._actions[action]
        curr_state = self._states[state.index]
        curr_valid = self._valid_actions[state.index]
        known_next = self._child_states[state.index]
        if action.index in known_next:
            return self.get_state(known_next[action.index], wrap=wrap)
        elif action.index not in curr_valid:
            for i in curr_valid:
                print(self._model._actions[i].name)
            raise RuntimeError(f'invalid action {action.name}')
        assigner = action.effect.resolve_in(curr_state) # TEST_HERE
        next_state = state.state_after(assigner)
        if self._vis_states[next_state.index] is None:
            next_vis_state = self._model.get_vis_after(state, next_state, action, self._vis_states[state.index])
            self._diffs_2.add((self.get_init_diff(state), self.get_init_diff(next_vis_state)))
            self._vis_states[next_state.index] = next_vis_state
            self._vis_encodings[next_state.index,:] = self._model.get_encoded(next_vis_state, no_cache=True)
            # self._vis_encodings[next_state.index] = self._model.get_encoded(next_vis_state, no_cache=True)
        known_next[action.index] = next_state.index
        return next_state if wrap else self._states[next_state.index]

class Describer:
    @classmethod
    def from_json(cls, world, json_data):
        return cls(world, json_data['verbs'], json_data['rules'])
    def __init__(self, world_model, verbs, rules):
        self._world = world_model
        self._verbs = verbs
        self._rules = rules
        self.context_keys = set().union(*(rule['context'].keys() for rule in rules))
    def get_rule(self, context):
        str_context = {key: (value if isinstance(value, str) else value.name) for key, value in context.items() if key in self.context_keys}
        for rule in self._rules:
            if all((str_context.get(key) == value for key, value in rule['context'].items())):
                return rule
        return {}
    def get_template(self, subject, context):
        if hasattr(subject, 'description'):
            default = subject.description
        elif hasattr(subject, 'name'):
            default = subject.name
        else:
            default = str(subject)
        return self.get_rule(context).get('template', default)
    def get_phrase(self, subject, to):
        context = subject.get_context() if hasattr(subject, 'get_context') else {}
        context['TO'] = to
        result = self.get_template(subject, context)
        for key, value in context.items():
            t_key = f'{{{key}}}'
            if t_key in result:
                result = result.replace(t_key, self.get_phrase(value, to))
        for key, tenses in self._verbs.items():
            index = result.find(key)
            while index != -1:
                replacement = tenses['third']
                if index >= 4 and result[index - 4:index - 1].lower() == 'you':
                    replacement = tenses['second']
                result = result[0:index] + replacement + result[index + len(key):]
                index = result.find(key)
        return result
    def get_sentence(self, subject, to):
        result = self.get_phrase(subject, to)
        if result is None or result == '':
            return result
        else:
            result = result[0:1].upper() + result[1:]
            if result[-1] != '.':
                result += '.'
            return result

class WorldModel:
    @classmethod
    def from_json(cls, json_data, cache_data=None):
        world = cls(json_data['name'])
        world.load_universe_from(json_data) # load domains from entities, variables, initial, and action_effects
        world.load_logic_from(json_data)
        # should be possible from here to optimize some of the conditions for faster testing
        world.do_optimizations()
        world.load_cache_from(cache_data)
        world.describer = Describer.from_json(world, json_data['describer'])
        return world
    def __init__(self, name):
        self.name = name
        self.FALSE = TT_value(-3, False, 'false', 'false')
        self.TRUE = TT_value(-2, True, 'true', 'true')
        self.NULL = TT_value(-1, None, 'null', 'null')
        self._constants = [self.FALSE, self.TRUE, self.NULL]
        self._cache = None
        self._instances = {}
        self._instance_data = {}
        # data (each entry is (identifiers, meta_data, logic_data))
        # identifiers always has an id and name, and has a code for non-variables
        # meta_data always has a description, has a signature for non-entities, and has an encoding for variables
        # self.entity_data = [] # logic_data has visibility
        # self.variable_data = [] # logic_data has visibility, initial_value
        # self.action_data = [] # logic_data has visibility, precondition, effect, consenting
        # self.ending_data = [] # logic_data has precondition
    def load_universe_from(self, json_data): # goal is to build out domains
        self._entities = []
        for i, entity in enumerate(json_data['entities']):
            if entity['type'] != 'Entity': raise TypeError(f'cannot load non-Entity type {entity["type"]} as Entity')
            self._entities.append(TT_value(*map(entity.__getitem__, ['id', 'name', 'name', 'description'])))
            self._entities[entity['id']]

        var_domain = {}
        initial_values = []
        for i, variable in enumerate(json_data['variables']):
            if variable['type'] != 'Variable': raise TypeError(f'cannot load non-Variable type {variable["type"]} as Variable')
            initial = {'type': 'Constant'} if json_data['initial'][i] is None else json_data['initial'][i]
            initial_values.append(self.get_tt_value(initial))
            var_domain[variable['id']] = {initial_values[variable['id']]}
        for i, effect in enumerate(json_data['action_effects']):
            for part in ([] if effect is None else effect):
                var_domain[part['variable']['id']].add(self.get_tt_value(part['value']))
        domain_to_var = {}
        for var_id, domain in var_domain.items():
            domain_to_var.setdefault(tuple(sorted(domain)), set()).add(var_id)
        var_domain.clear()
        self._domains = []
        for domain, var_ids in domain_to_var.items():
            encodings = np.eye(len(domain), dtype=np.bool)
            constants = tuple(filter(lambda v: v.is_constant, domain))
            self._domains.append(TT_domain(len(self._domains), domain, constants, encodings, tuple(sorted(var_ids)),
                domain == (self.FALSE, self.TRUE) ))
            for var_id in var_ids:
                var_domain[var_id] = self._domains[-1].index
        offset = 0
        self._variables = []
        for i, variable in enumerate(json_data['variables']):
            sig = self.get_tt_signature(variable['signature'])
            width = len(self._domains[var_domain[variable['id']]].values)
            self._variables.append(TT_variable(*map(variable.__getitem__, ['id', 'name', 'description']),
                var_domain[variable['id']], initial_values[i], sig, offset, width))
            offset += width
            self._variables[variable['id']]
        self._values = self._entities + self._constants
        self.PC = self._entities[0]
        initial_facts = list(map(lambda v: self.get_tt_fact(v, v.initial), self._variables))
        self._initial = TT_state(initial_facts)
        self._say_once = set()
        self.state_encoding_width = sum([var.width for var in self._variables])
    def load_logic_from(self, json_data):
        self._var_visibility = []
        for i, variable in enumerate(json_data['variables']):
            visibility = self.process_tt_condition(json_data['variable_visibility'][i], outer=True)
            self._var_visibility.append(visibility)
        self._actions = []
        for i, action in enumerate(json_data['actions']):
            effect = self.process_tt_effect(json_data['action_effects'][i])
            precondition = self.process_tt_condition(json_data['action_preconditions'][i], outer=True)
            visibility = self.process_tt_condition(json_data['action_visibility'][i], outer=True)
            sig = self.get_tt_signature(action['signature'])
            consenting = tuple(map(self.get_tt_value, action['consenting']))
            player_action = self.PC in consenting
            system_action = not self.PC in consenting or len(consenting) > 1
            self._actions.append(TT_action(*map(action.__getitem__, ['id', 'name', 'description']),
                sig, precondition, effect, visibility, consenting, player_action, system_action))
        self._endings = []
        for i, ending in enumerate(json_data['endings']):
            precondition = self.process_tt_condition(json_data['ending_conditions'][i], outer=True)
            self._endings.append(TT_ending(*map(ending.__getitem__, ['id', 'name', 'description']), sig, precondition))
    def do_optimizations(self):
        equality_tests = set()
        for condition in self._var_visibility:
            equality_tests |= condition.get_equality_tests()
        for action in self._actions:
            equality_tests |= action.get_equality_tests()
        for ending in self._endings:
            equality_tests |= ending.get_equality_tests()
        self._stored_tests = []
        self._test_refs = {}
        for test in equality_tests:
            self._test_refs[test] = TT_test_reference(len(self._stored_tests), self)
            self._stored_tests.append(test)
        # for i, condition in enumerate(self._var_visibility):
        #     self._var_visibility[i] = condition.replace_tests(self._test_refs)
        # for i, action in enumerate(self._actions):
        #     self._actions[i] = action.replace_tests(self._test_refs)
        # for i, ending in enumerate(self._endings):
        #     self._endings[i] = ending.replace_tests(self._test_refs)
    def get_tested_state(self, state):
        return tuple(map(methodcaller('test_in', state), self._stored_tests))
    def load_cache_from(self, cache_data=None):
        self._cache = WorldCache(self)
    def get_tt_value(self, entry):
        if entry['type'] == 'Constant':
            return self.NULL if entry.get('value', None) is None else (self.TRUE if entry['value'] else self.FALSE)
        elif entry['type'] == 'Entity':
            return self._entities[entry['id']]
        else:
            raise TypeError(f'invalid type: {entry["type"]}')
    def get_tt_action(self, index):
        return self._actions[index]
    def get_tt_signature(self, entry):
        return TT_signature(entry['name'], tuple(map(self.get_tt_value, entry['arguments'])))
    def get_tt_fact(self, variable, value, throw=False):
        flag = ('fact', variable, value)
        domain = self._domains[variable.domain_id]
        if value not in domain.values:
            if throw:
                raise RuntimeError(f'{value} is not in the domain of {variable}, so no fact exists')
            flag = (value, variable)
            if flag not in self._say_once:
                logging.info(f'{value} is not in the domain of {variable}, so no fact exists')
                self._say_once.add(flag)
            return None
        internal_id = domain.values.index(value)
        fact = domain.get_fact(value)
        return fact
    def process_tt_condition(self, entry, outer=False):
        if outer:
            condition = self.process_tt_condition(entry)
            if isinstance(condition, TT_value):
                if condition == self.TRUE:
                    condition = Conjunction()
                elif condition == self.FALSE:
                    condition = Disjunction()
                else:
                    raise RuntimeError(f'unexpected value: {condition}')
            return condition
        if entry is None: entry = {'type': 'Constant', 'value': True}
        if entry['type'] in {'Constant', 'Entity'}:
            return self.get_tt_value(entry)
        elif entry['type'] == 'Variable':
            return self._variables[entry['id']]
        elif entry['type'] == 'Proposition' and entry['operator'] == 'EQUALS':
            if len(entry['arguments']) != 2: raise TypeError(f'invalid number of arguments to EQUALS: {len(entry["arguments"])}')
            lhs, rhs = tuple((self.process_tt_condition(a) for a in  entry['arguments']))
            if isinstance(lhs, TT_value) and isinstance(rhs, TT_value):
                return self.TRUE if lhs == rhs else self.FALSE
            elif isinstance(lhs, TT_value) and isinstance(rhs, TT_variable):
                fact = self.get_tt_fact(rhs, lhs)
                return self.FALSE if fact is None else TT_equality(rhs, fact)
            elif isinstance(lhs, TT_variable) and isinstance(rhs, TT_value):
                fact = self.get_tt_fact(lhs, rhs)
                return self.FALSE if fact is None else TT_equality(lhs, fact)
            else: # both TT_variable
                if lhs.index > rhs.index: lhs, rhs = rhs, lhs
                return self.TRUE if lhs == rhs else TT_equality(lhs, rhs)
        elif entry['type'] == 'Proposition' and entry['operator'] in {'AND', 'OR'}:
            processed = tuple((self.process_tt_condition(a) for a in  entry['arguments']))
            parts = []
            for arg in processed:
                if isinstance(arg, TT_variable):
                    fact = self.get_tt_fact(arg, self.TRUE, throw=True)
                    parts.append(TT_equality(arg, fact))
                elif isinstance(arg, TT_value):
                    if arg == self.FALSE:
                        if entry['operator'] == 'AND': return Disjunction()
                    elif arg == self.TRUE:
                        if entry['operator'] == 'OR': return Conjunction()
                    else:
                        raise RuntimeError(f'no way to handle {arg}')
                elif entry['operator'] == 'AND' and isinstance(arg, Conjunction):
                    parts.extend(arg.args)
                elif entry['operator'] == 'OR' and isinstance(arg, Disjunction):
                    parts.extend(arg.args)
                else:
                    parts.append(arg)
            if len(parts) == 1:
                return parts[0]
            elif entry['operator'] == 'AND':
                return Conjunction(tuple(parts))
            else:
                return Disjunction(tuple(parts))
        elif entry['type'] == 'Proposition' and entry['operator'] == 'NOT':
            if len(entry['arguments']) != 1: raise TypeError(f'invalid number of arguments to NOT: {len(entry["arguments"])}')
            arg = self.process_tt_condition(entry['arguments'][0])
            if isinstance(arg, TT_value):
                return self.FALSE if arg.value else self.TRUE
            elif isinstance(arg, Disjunction) and len(arg.args) == 0:
                return Conjunction()
            elif isinstance(arg, TT_variable):
                fact = self.get_tt_fact(arg, self.TRUE, throw=True)
                return Negation(TT_equality(arg, fact))
            else:
                return Negation(arg)
        elif entry['type'] == 'Proposition':
            raise NotImplementedError(f'unrecognized condition operator: {entry["operator"]}')
        else:
            raise NotImplementedError(f'unrecognized condition type: {entry["type"]}')
    def process_tt_effect(self, entry):
        if entry is None: entry = []
        unconditional = []
        conditional = []
        requirements = []
        for i, part in enumerate(entry):
            variable = self._variables[part['variable']['id']]
            value = self.get_tt_value(part['value'])
            fact = self.get_tt_fact(variable, value, throw=True)
            condition = self.process_tt_condition(part.get('condition', {'type': 'Proposition', 'operator': 'AND', 'arguments': []}), outer=True)
            if isinstance(condition, Conjunction) and len(condition.args) == 0:
                unconditional.append((variable, fact))
            else:
                conditional.append(TT_assign((variable,), (fact,)))
                requirements.append(condition)
        always = TT_assign(*zip(*unconditional)) if unconditional else TT_assign((), ())
        return TT_effect(always, conditional, requirements)
    def get_tt_actions_in(self, state, no_cache=False):
        if no_cache or self._cache is None:
            return [act for act in self._actions if act.precondition.test_in(state)] # TEST_HERE
        else:
            act_indexes = self._cache._valid_actions[state.index]
            return [self._actions[i] for i in act_indexes]
    def get_tt_endings_in(self, state, no_cache=False):
        if no_cache or self._cache is None:
            return [end for end in self._endings if end.precondition.test_in(state)] # TEST_HERE
        else:
            return self._cache._endings[state.index]
    def get_encoded(self, state, no_cache=False):
        if no_cache or self._cache is None:
            parts = []
            for var in self._variables:
                domain = self._domains[var.domain_id]
                parts.append(domain.encodings[state.facts[var.index].code_index])
            return np.concat(parts)
        else:
            return self._cache._encodings[state.index,:]
    def get_initial(self):
        if self._cache is None:
            return self._initial
        else:
            return self._cache.get_state(0)
    def get_vis_after(self, before, after, action, vis_before, no_cache=False):
        # if action was visible in before, its assignment is part of the update
        # anything visible in after is also part of the update
        visible = action.visibility.test_in(before) # TEST_HERE
        assigner = action.effect.resolve_in(before) if visible else TT_assign((), ())
        update_vars = []
        update_facts = []
        for var, visibility in zip(self._variables, self._var_visibility):
            if (var in assigner.variables or visibility.test_in(after)): # TEST_HERE
                update_vars.append(var)
                update_facts.append(after.facts[var.index])
        vis_assign = TT_assign(tuple(update_vars), tuple(update_facts))
        return vis_before.state_after(vis_assign)
    def get_random_story(self, max_length=None):
        story = self.get_empty_story()
        valid_actions, valid_endings = self.get_valid_events(story, max_length=max_length)
        while len(valid_actions):
            action = choice(valid_actions)
            self.update_story(story, action, _checked=True)
            valid_actions, valid_endings = self.get_valid_events(story, max_length=max_length)
        if not valid_endings:
            story.events.append(None)
        return story
    def get_empty_story(self):
        return TT_story([self.get_initial()], [self.get_initial()], [])
    def get_valid_events(self, story, max_length=None):
        endings = self.get_tt_endings_in(story.world_states[-1])
        if max_length is not None and len(story.events) >= max_length:
            actions = []
        actions = [] if endings else self.get_tt_actions_in(story.world_states[-1])
        return actions, endings
    def update_story(self, story, action, _checked=False):
        if _checked:
            story.events.append(action)
            next_ws = self._cache.get_next_state(story.world_states[-1], story.events[-1])
            next_vs = self._cache._vis_states[next_ws.index]
            story.world_states.append(next_ws)
            story.visible_states.append(next_vs)
            valid_endings = self.get_tt_endings_in(next_ws)
            if valid_endings:
                story.events.append(valid_endings[0])
        else:
            actions, endings = self.get_valid_events(story)
            if action not in actions:
                raise ValueError(f'{action} is not valid in {story}')
            return self.update_story(story, action, _checked=True)


# class Discovery:
#     def __init__(self, world, effect):
#         self.world = world
#         self.eff = effect
#     def apply_to(self, state):
#         return state.get_state_after(self.eff)

# class StoryModel:
#     def __init__(self, world_model):
#         self._world = world_model
#     def get_state(self, index, use_vis=False):
#         return self._vis_states[index] if use_vis else self._states[index]
#     def get_actions(self, index, use_vis=False):
#         return self._vis_actions[index] if use_vis else self._actions[index]
#     def get_raw_actions(self, index, use_vis=False):
#         return self._raw_vis_actions[index] if use_vis else self._actions[index]
#     def current_state(self, use_vis=False):
#         return self.get_state(-1, use_vis=use_vis)
#     def current_actions(self, use_vis=False):
#         return self.get_actions(-1, use_vis=use_vis)
#     def current_raw_actions(self, use_vis=False):
#         return self.get_raw_actions(-1, use_vis=use_vis)
#     def reset(self):
#         self._states = [State(self._world, tuple([var.initial.value for var in self._world.variables]))]
#         self._actions = [self._world.get_actions_in(self._states[-1])]
#         self._endings = self._world.get_endings_in(self._states[-1])
#         self._vis_states = self._states[:]
#         self._vis_actions = [self._world.get_actions_in(self._states[-1], self._vis_states[-1])]
#         self._raw_vis_actions = self._actions[:]
#         self._history = [None]
#         self._vis_discoveries = [Discovery(self._world, Effect(self._world, ()))]
#     def update(self, action):
#         if self._endings:
#             raise RuntimeError(f'cannot update a finished story ({", ".join(map(str, self._endings))})')
#         before = self._states[-1]
#         after = action.apply_to(before)
#         self._states.append(after)
#         self._actions.append(self._world.get_actions_in(after))
#         self._endings = self._world.get_endings_in(after)
#         vis_before = self._vis_states[-1]
#         vis_after = action.apply_to(vis_before, assume_valid=True) if action.visible_in(before) else vis_before
#         vis_updates = [assign for assign in after.get_visible() if assign.value.value != vis_after.get_value_at(assign.var.index)]
#         discovery = Discovery(self._world, Effect(self._world, tuple(vis_updates)))
#         self._vis_states.append(discovery.apply_to(vis_after))
#         # determine actions visibly possible in the player perspective
#         self._raw_vis_actions.append(self._world.get_actions_in(self._vis_states[-1]))
#         self._vis_actions.append(self._world.get_actions_in(self._states[-1], self._vis_states[-1]))
#         # self._vis_actions.append(self._world.get_actions_in(self._vis_states[-1], visible=True))
#         self._history.append(action)
#         self._vis_discoveries.append(discovery)

# class Command:
#     def __init__(self, action, ctype, force=False):
#         self.action = action
#         self.ctype = ctype
#         self.turn = 'OFFER' if ctype == 'PROPOSE' or force else 'REPLY'
#     def as_tuple(self):
#         return (self.ctype, self.turn, '' if self.action is None else self.action.name)
#     def __str__(self):
#         if self.action is None:
#             return self.ctype
#         else:
#             return f'{self.ctype:7} {self.action}'
#     def __repr__(self):
#         return str(self)
#     def __eq__(self, other):
#         return isinstance(other, Command) and self.action == other.action and self.ctype == other.ctype and self.turn == other.turn
#     def __hash__(self):
#         return hash((self.action, self.ctype, self.turn))

# class Turn:
#     def __init__(self, role, cmd, index):
#         self.role = role
#         self.ttype = cmd.ctype
#         self.action = cmd.action
#         self.story_index = index
#     def get_context(self):
#         result = {'ROLE': self.role, 'TURN_TYPE': self.ttype}
#         if not self.action is None:
#             result.update(self.action.get_context())
#         return result

# class OldSystemView:
#     def __init__(self, game):
#         self._game = game
#         self._role = game.SYS_NAME
#         self._world = game._world
#         self._partial_view = None
#     @property
#     def role_actions(self):
#         return self._game.role_actions[self._role]
#     @property
#     def role_proposal_ids(self):
#         return self._game.role_proposal_ids[self._role]
#     def reset(self):
#         self._turn_history = self._game._turn_history
#         self.turn_index = 0
#         self._story = self._game._story
#     def option_indexes(self, use_vis=False): # use_vis is ignored
#         if self._game.finished or self._game.actor != self._role: return []
#         command_indexes = []
#         for act in self._story.current_actions():
#             command_indexes += list(self._game.story_act_index[act][self._role])
#         if self._game._offered is None:
#             return [0] + [i for i in command_indexes if self.role_actions[i].turn == 'OFFER']
#         else:
#             return [i for i in command_indexes if self.role_actions[i].turn == 'REPLY'
#                 and self._game._offered == self.role_actions[i].action]
#     def list_options(self, use_vis=False):
#         return [self.role_actions[i] for i in self.option_indexes(use_vis=use_vis)]
#     def display_turns(self, indent=''):
#         while self.turn_index < len(self._turn_history):
#             turn_entry = self._turn_history[self.turn_index]
#             if isinstance(turn_entry, Turn) and turn_entry.ttype == 'PASS' and turn_entry.role == self._role:
#                 pass
#             else:
#                 described = self._world.describer.get_sentence(turn_entry, self._role)
#                 print(f'{indent} {described}')
#             self.turn_index += 1
#     def view_data(self):
#         turn_field_width = 2
#         offer_field_width = len(self._game.player_view().role_proposal_ids)
#         state_field_width = len(self._story.get_state(0).get_onehot())
#         action_count = len(self._world.actions)
#         return turn_field_width, offer_field_width, state_field_width, action_count
#     def turn_data(self, use_vis=False):
#         turn_field_width = 2
#         turn_field_index = 0 if self._game._turn == 'OFFER' else 1
#         offer_field_width = len(self._game.player_view().role_proposal_ids)
#         offer_field_index = None if self._game._offered is None else self._game.player_view().role_proposal_ids[self._game._offered.index]
#         turn_vector = [0] * turn_field_width
#         turn_vector[turn_field_index] = 1
#         offer_vector = [0] * offer_field_width
#         if not offer_field_index is None: offer_vector[offer_field_index] = 1
#         current_state = self._story.current_state(use_vis=use_vis)
#         state_vector = current_state.get_onehot()
#         valid_actions = self._story.current_raw_actions(use_vis=use_vis)
#         return tuple(turn_vector), tuple(offer_vector), state_vector, valid_actions 

# class OldPlayerView:
#     def __init__(self, game):
#         self._game = game
#         self._role = game.PLA_NAME
#         self._world = game._world
#         self._partial_view = {var for var in self._world.variables if not var.vis.trivially_false()}
#         self._ignore_indexes = {var.index for var in self._world.variables if var.vis.trivially_false()}
#     @property
#     def role_actions(self):
#         return self._game.role_actions[self._role]
#     @property
#     def role_proposal_ids(self):
#         return self._game.role_proposal_ids[self._role]
#     def reset(self):
#         self._turn_history = []
#         self.turn_index = 0
#         self._story = self._game._story
#         self.displayed_discoveries = set()
#     def option_indexes(self, use_vis=True):
#         if self._game.finished or self._game.actor != self._role: return []
#         command_indexes = []
#         for act in self._story.current_actions(use_vis=use_vis):
#             command_indexes += list(self._game.story_act_index[act][self._role])
#         if self._game._offered is None:
#             return [0] + [i for i in command_indexes if self.role_actions[i].turn == 'OFFER']
#         else:
#             return [i for i in command_indexes if self.role_actions[i].turn == 'REPLY'
#                 and self._game._offered == self.role_actions[i].action]
#     def list_options(self, use_vis=True):
#         return [self.role_actions[i] for i in self.option_indexes(use_vis=use_vis)]
#     def display_turns(self, indent=''):
#         while self.turn_index < len(self._turn_history):
#             turn_entry = self._turn_history[self.turn_index]
#             if isinstance(turn_entry, Turn) and turn_entry.ttype in {'PASS', 'PROPOSE'} and turn_entry.role == self._role:
#                 pass
#             elif isinstance(turn_entry, Turn):
#                 turn_i = self.turn_index
#                 story_i = turn_entry.story_index
#                 described = self._world.describer.get_sentence(turn_entry, self._role)
#                 print(f'[{turn_i:>3},{story_i:>3}] {indent} {described}')
#                 if story_i not in self.displayed_discoveries:
#                     discovery = self._story._vis_discoveries[story_i]
#                     if discovery.eff.args:
#                         sentences = list(filter(lambda s: len(s) > 0, [self._world.describer.get_sentence(assign, self._role) for assign in discovery.eff.args]))
#                         if sentences:
#                             print((' '*11) + indent + '-  ' + " ".join(sentences))
#                     self.displayed_discoveries.add(story_i)
#             else:
#                 described = self._world.describer.get_sentence(turn_entry, self._role)
#                 print(f'{" "*10}{indent} {described}')
#             self.turn_index += 1
#     def view_data(self):
#         turn_field_width = 2
#         offer_field_width = len(self._game.system_view().role_proposal_ids)
#         state_field_width = len(self._story.get_state(0).get_onehot(self._ignore_indexes))
#         action_count = len(self._world.actions)
#         return turn_field_width, offer_field_width, state_field_width, action_count
#     def turn_data(self, use_vis=True):
#         turn_field_width = 2
#         turn_field_index = 0 if self._game._turn == 'OFFER' else 1
#         offer_field_width = len(self._game.system_view().role_proposal_ids)
#         offer_field_index = None if self._game._offered is None else self._game.system_view().role_proposal_ids[self._game._offered.index]
#         turn_vector = [0] * turn_field_width
#         turn_vector[turn_field_index] = 1
#         offer_vector = [0] * offer_field_width
#         if not offer_field_index is None: offer_vector[offer_field_index] = 1
#         current_state = self._story.current_state(use_vis=use_vis)
#         state_vector = current_state.get_onehot(self._ignore_indexes)
#         valid_actions = self._story.current_raw_actions(use_vis=use_vis)
#         return tuple(turn_vector), tuple(offer_vector), state_vector, valid_actions 

# SYSTEM_OPTS = [actions without PC] x [SUCCEED, FAIL] + [actions with PC and con > 1] x [PROPOSE, SUCCEED, FAIL] + [actions with PC and con = 1] x [SUCCEED, FAIL] + [PASS]
# PLAYER_OPTS = [actions with PC] x [PROPOSE, SUCCEED, FAIL] + [PASS]
# class OldGameModel:
#     @property
#     def actor(self):
#         return self._roles[0]
#     @property
#     def partner(self):
#         return self._roles[1]
#     def flip_turn(self):
#         self._roles = tuple(reversed(self._roles))
#     def get_turn(self):
#         return f'{self._turn}({self.actor})'
#     def __init__(self, world_model):
#         self.SYS_NAME = 'GAME_MASTER'
#         self.PLA_NAME = 'PLAYER'
#         self.PASS = Command(None, 'PASS')
#         self.PC = world_model.get_entity(0)
#         self._world = world_model
#         self._story = StoryModel(world_model)
#         self.role_views = {self.SYS_NAME: SystemView(self), self.PLA_NAME: PlayerView(self)}
#         self.role_actions = {self.SYS_NAME: [self.PASS], self.PLA_NAME: [self.PASS]}
#         self.story_act_index = {}
#         self.role_proposal_ids = {self.SYS_NAME: {}, self.PLA_NAME: {}}
#         for act in self._world.actions:
#             system_act_count = len(self.role_actions[self.SYS_NAME])
#             player_act_count = len(self.role_actions[self.PLA_NAME])
#             if self.PC in act.con:
#                 self.role_actions[self.PLA_NAME] += [Command(act, mod) for mod in ('PROPOSE', 'SUCCEED', 'FAIL')]
#                 ctypes = ('SUCCEED', 'FAIL') if len(act.con) == 1 else ('PROPOSE', 'SUCCEED', 'FAIL')
#                 self.role_actions[self.SYS_NAME] += [Command(act, mod) for mod in ctypes]
#                 self.role_proposal_ids[self.PLA_NAME][act.index] = len(self.role_proposal_ids[self.PLA_NAME])
#                 if len(act.con) != 1:
#                     self.role_proposal_ids[self.SYS_NAME][act.index] = len(self.role_proposal_ids[self.SYS_NAME])
#             else:
#                 self.role_actions[self.SYS_NAME] += [Command(act, mod, True) for mod in ('SUCCEED', 'FAIL')]
#                 # self.role_proposal_ids[self.SYS_NAME][act.index] = len(self.role_proposal_ids[self.SYS_NAME])
#             self.story_act_index[act] = {
#                 self.PLA_NAME: set(range(player_act_count, len(self.role_actions[self.PLA_NAME]))),
#                 self.SYS_NAME: set(range(system_act_count, len(self.role_actions[self.SYS_NAME])))}
#         self.reset()
#     def role_names(self):
#         return (self.SYS_NAME, self.PLA_NAME)
#     def system_view(self):
#         return self.role_views[self.SYS_NAME]
#     def player_view(self):
#         return self.role_views[self.PLA_NAME]
#     def reset(self):
#         self._story.reset()
#         self._roles = (self.SYS_NAME, self.PLA_NAME)
#         self._turn = 'OFFER'
#         self._offered = None
#         self._turn_history = []
#         self._command_history = []
#         for role_view in self.role_views.values(): role_view.reset()
#     @property
#     def finished(self):
#         return len(self._story._endings) > 0
#     def do_command(self, command):
#         self._command_history.append((self.actor, command))
#         self.record_command(command)
#         if command.ctype == 'PASS':
#             if self._turn != 'OFFER': raise RuntimeError(f'can only PASS on an OFFER turn, not {self._turn}')
#             self._roles = tuple(reversed(self._roles))
#         elif command.ctype == 'PROPOSE':
#             if self._turn != 'OFFER': raise RuntimeError(f'can only PROPOSE on an OFFER turn, not {self._turn}')
#             if command.action not in self._story._actions[-1]:
#                 print('valid options:')
#                 for act in self._story._actions[-1]:
#                     print(act)
#                 raise RuntimeError(f'can only PROPOSE a valid action in the current state, not {command.action}')
#             self._roles = tuple(reversed(self._roles))
#             self._offered = command.action
#             self._turn = 'REPLY'
#         else:
#             if command.action not in self._story._actions[-1]: raise RuntimeError(f'can only {command.ctype} a valid action in the current state, not {command.action}')
#             if self._turn == 'REPLY' and command.action != self._offered: raise RuntimeError(f'can only {command.ctype} the offered action ({self._offered}) on a REPLY turn, not {command.action}')
#             if command.ctype == 'SUCCEED':
#                 self._story.update(command.action)
#             if self._story._endings:
#                 self.record_ending(self._story._endings[0])
#             if self._turn == 'REPLY':
#                 self._turn = 'OFFER'
#                 self._offered = None
#                 # when the player responds to an offer, control returns to the system
#                 if self.actor == self.PLA_NAME:
#                     self._roles = tuple(reversed(self._roles))
#                 # when the system responds to an offer, it is also the next to offer
#     def record_command(self, command, force_record=True):
#         story_i = len(self._story._states) if command.ctype in {'SUCCEED'} else len(self._story._states) - 1
#         entry = Turn(self.actor, command, story_i)
#         self._turn_history.append(entry)
#         if command.ctype == 'PASS' or command.action.visible_in(self._story.current_state()) or self._turn == 'REPLY' and self.actor == self.SYS_NAME:
#             self.role_views[self.PLA_NAME]._turn_history.append(entry)
#         elif force_record:
#             self.role_views[self.PLA_NAME]._turn_history.append(entry)
#     def record_ending(self, ending):
#         self._turn_history.append(ending)
#         self.role_views[self.PLA_NAME]._turn_history.append(ending)

# rework plans:
#  each variable is associated with a width (of the one-hot vector)
#  a (variable, value) pair is associated with an index
#  states are represented as a concatenation of one-hot vectors
#  each variable has a start index and an end index, in that range one value is hot
#
#  comparisons can be precompiled
#  healthy(Adventurer) spans expanded-state indexes [10, 11, 12]
#  healthy(Adventurer) == Dead: s[10:13] == [1, 0, 0]
#  healthy(Adventurer) == Healthy: s[10:13] == [0, 1, 0]
#  healthy(Adventurer) == Hurt: s[10:13] == [0, 0, 1]
#  or... s[10+0] == 1, s[10+1] == 1, s[10+2] == 1
#  
#  location(Adventurer) is at s[109:114], location(Alchemist) is at s[114:119]
#  location(Adventurer) == location(Alchemist) if bool(s[109:114] & s[114:119]) 
#
#  conjunctions can be represented as a mask, disjunctions can be spread over rows
#

def load_world_model(path):
    if isinstance(path, str): path = Path(path)
    with path.open('r') as fp:
        json_data = json.load(fp)
    return WorldModel.from_json(json_data)

if __name__ == '__main__':
    from random import choice, randint
    def main(args):
        world_model = load_world_model(args.model_file)
        print(world_model.name)
        # world_model.show_entities()
        # world_model.show_variables()
        # world_model.show_actions(10)
        # world_model.show_endings()
        from collections import Counter
        endings = Counter()
        # for i in range(100):
        #     failed, count = world_model.do_world_test()
        #     if failed:
        #         raise RuntimeError()
        step_count = 0
        overlapping = Counter()
        for i in range(15000):
            story = world_model.get_random_story(max_length=100)
            overlapping[tuple(map(attrgetter('index'), story.events[:-5]))] += 1
            endings[story.events[-1]] += 1
            step_count += len(story.events) - 1
            if i % 2000 == 0:
                logging.info(f'\t[{i:>4}] {step_count/(i+1)} steps on average')
                for ending in endings:
                    logging.info(f'\t{ending} occurred {endings[ending]} times')
                print(f'{sum(overlapping.values())} generated')
                print(f'{sum(filter(lambda x: x > 1, overlapping.values()))} overlapped')
        logging.info(f'{step_count/(i+1)} steps on average')
        if world_model._cache is not None:
            print(world_model._cache)
        game = GameModel(world_model)
        return 0

    parser = ArgumentParser(prog='TestTandemTales',
                        description='A program which tests Tandem Tales interpretation tools.')
    parser.add_argument('model_file', help='A JSON file describing the story model that should be used.')
    parser.add_argument('-l', '--loglevel', metavar='LEVEL', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='WARNING',
        help='The logging verbosity level, selected from DEBUG, INFO, WARNING, ERROR, or CRITICAL.', type=str.upper)
    parser.add_argument('-L', '--log', default='stderr', help='The name of a file or stream to log to.', type=str.lower)
    parser.add_argument('--profile', action='store_true', help='A flag indicating that main should be run with profiling turned on.')

    args = parser.parse_args()

    if args.log in {'stderr', '-', 'stdout'}:
        logging.basicConfig(
            format='[%(asctime)s] (%(levelname)s|%(name)s|%(funcName)s) %(message)s', datefmt='%H:%M:%S',
            level=getattr(logging, args.loglevel.upper()),
            stream=sys.stdout if args.log == 'stdout' else sys.stderr,
        )
    else:
        logging.basicConfig(
            format='[%(asctime)s] (%(levelname)s|%(name)s|%(funcName)s) %(message)s', datefmt='%H:%M:%S',
            level=getattr(logging, args.loglevel.upper()),
            filename=args.log
        )
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
        statistics.print_stats(.01)
        statistics = pstats.Stats(profiler, stream=stream).sort_stats(SortKey.TIME)
        statistics.print_stats(.01)
        print(stream.getvalue())
        statistics.dump_stats('profile.out')

    exit(result)
