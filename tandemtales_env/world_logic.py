import logging
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
        if isinstance(value, TT_value):
            if value not in self.values:
                return None
            internal_id = self.values.index(value)
        else:
            internal_id = value
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
class TT_equality(Proposition):
    lhs : TT_variable
    rhs : TT_variable|TT_fact
    def __str__(self):
        return ' == '.join(map(str, (self.lhs, self.rhs)))
    def test_in(self, state):
        l_fact = state.facts[self.lhs.index]
        r_fact = state.facts[self.rhs.index] if isinstance(self.rhs, TT_variable) else self.rhs
        return l_fact == r_fact
        # if l_fact.domain_id != r_fact.domain_id:
        #     raise RuntimeError('this should not be possible')
        #     l_fact, r_fact = l_fact.value, r_fact.value
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
        return f'{self.variable} := {self.fact}'
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
    def is_empty(self):
        return len(self.facts) == 0
    def get_indexes(self):
        return list(map(attrgetter('index'), self.variables)), list(map(attrgetter('code_index'), self.facts))

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
class TT_np_state:
    vector : Sequence[int]
    world : Any
    @property
    def facts(self):
        return self.world.get_state_facts(self.vector)
    def state_after(self, assigner):
        v_is, f_is = assigner.get_indexes()
        new_vector = self.vector.copy()
        new_vector[v_is] = f_is
        return TT_np_state(new_vector, self.world)

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
    def get_var_equality_pairs(self, entry):
        if entry is None: entry = {'type': 'Constant', 'value': True}
        if isinstance(entry, list): # effect
            pairs = set()
            for part in entry:
                if 'condition' in part:
                    result = self.get_var_equality_pairs(part['condition'])
                    if isinstance(result, set):
                        pairs |= result
            return pairs
        elif entry['type'] in {'Constant', 'Entity'}:
            return self.get_tt_value(entry)
        elif entry['type'] == 'Variable':
            return entry['id']
        elif entry['type'] == 'Proposition' and entry['operator'] == 'EQUALS':
            if len(entry['arguments']) != 2: raise TypeError(f'invalid number of arguments to EQUALS: {len(entry["arguments"])}')
            lhs, rhs = tuple((self.get_var_equality_pairs(a) for a in  entry['arguments']))
            pairs = set()
            if isinstance(lhs, int) and isinstance(rhs, int):
                pairs.add((lhs, rhs))
            return pairs
        elif entry['type'] == 'Proposition':
            processed = tuple((self.get_var_equality_pairs(a) for a in  entry['arguments']))
            pairs = set()
            for arg in entry['arguments']:
                result = self.get_var_equality_pairs(arg)
                if isinstance(result, set):
                    pairs |= result
            return pairs
        else:
            raise NotImplementedError(f'unrecognized condition type: {entry["type"]}')
    def merge_equality_domains(self, json_data, var_domain):
        pairs = set()
        for i, variable in enumerate(json_data['variables']):
            result = self.get_var_equality_pairs(json_data['variable_visibility'][i])
            if isinstance(result, set): pairs |= result
        for i, action in enumerate(json_data['actions']):
            result = self.get_var_equality_pairs(json_data['action_effects'][i])
            if isinstance(result, set): pairs |= result
            result = self.get_var_equality_pairs(json_data['action_preconditions'][i])
            if isinstance(result, set): pairs |= result
            result = self.get_var_equality_pairs(json_data['action_visibility'][i])
            if isinstance(result, set): pairs |= result
        for i, ending in enumerate(json_data['endings']):
            result = self.get_var_equality_pairs(json_data['ending_conditions'][i])
            if isinstance(result, set): pairs |= result
        for vi1, vi2 in pairs:
            merged = var_domain[vi1] | var_domain[vi2]
            if merged != var_domain[vi1]: var_domain[vi1] |= merged
            if merged != var_domain[vi2]: var_domain[vi2] |= merged
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
        self.merge_equality_domains(json_data, var_domain)
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
        self.fact_assignments = []
        for var in self._variables:
            domain = self._domains[var.domain_id]
            for value in domain.values:
                fact = domain.get_fact(value)
                self.fact_assignments.append(TT_assignment(var, fact))
        initial_fcodes = list(map(lambda v: self.get_tt_fact(v, v.initial).code_index, self._variables))
        # self._initial = TT_np_state(np.array(initial_fcodes), self)
        self._say_once = set()
        self.state_encoding_width = sum([var.width for var in self._variables])
    def load_logic_from(self, json_data):
        self.never_visible = []
        self._var_visibility = []
        for i, variable in enumerate(json_data['variables']):
            visibility = self.process_tt_condition(json_data['variable_visibility'][i], outer=True)
            self._var_visibility.append(visibility)
            self.never_visible.append(visibility.trivially_false())
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
    def get_state_facts(self, domain_indices):
        facts = []
        for variable in self._variables:
            code_index = domain_indices[variable.index]
            fact = self._domains[variable.domain_id].get_fact(code_index)
            facts.append(fact)
        return facts
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
    def get_encoded(self, state, vis=False, no_cache=False):
        if no_cache or self._cache is None:
            parts = []
            for var in self._variables:
                domain = self._domains[var.domain_id]
                parts.append(domain.encodings[state.facts[var.index].code_index])
            return np.concat(parts)
        elif vis:
            return self._cache._vis_encodings[state.index,:]
        else:
            return self._cache._encodings[state.index,:]
    def get_initial(self):
        if self._cache is None:
            return self._initial
        else:
            return self._cache.get_state(0)
    def get_vis_changes(self, before, after, action, vis_before, no_cache=False):
        visible = action.visibility.test_in(before)
        act_assign = action.effect.resolve_in(before) if visible else TT_assign((), ())
        update_vars = []
        update_facts = []
        for var, visibility in zip(self._variables, self._var_visibility):
            if var not in act_assign.variables and visibility.test_in(after) and vis_before.facts[var.index] != after.facts[var.index]:
                update_vars.append(var)
                update_facts.append(after.facts[var.index])
        discover_assign = TT_assign(tuple(update_vars), tuple(update_facts))
        return visible, act_assign, discover_assign
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
            #next_vs = self._cache._vis_states[next_ws.index]
            next_vs = self._cache.get_state(next_ws.index, vis=True)
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
    def get_vis_update(self, story, index=None):
        if index is None: index = len(story.world_states) - 2
        return self.get_vis_changes(story.world_states[index], story.world_states[index+1], story.events[index], story.visible_states[index])

def load_world_model(path):
    if isinstance(path, str): path = Path(path)
    with path.open('r') as fp:
        json_data = json.load(fp)
    return WorldModel.from_json(json_data)