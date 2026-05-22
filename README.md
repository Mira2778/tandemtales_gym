# TandemTalesEnv - RL for Tandem Tales via Gymnasium

[Tandem Tales](https://github.com/sgware/tt-server) is a platform for playing
paired, text-based interactive storytelling games involving a player who makes
choices for one character and a game master that makes all other choices in the
narrative.

TandemTalesEnv is a [Gymnasium](https://gymnasium.farama.org/) environment
which models the cooperative storytelling experience of Tandem Tales as a
single-player game played by one role or the other. (The actions of the absent
partner role are reinterpreted as nondeterministic transition behavior.)

The behavior of TandemTalesEnv games is highly configurable, first and
foremost by the choice of the Tandem Tales story world. Additional control over
the environment is facilitated by modular components that determine the
behavior of the partner agent and the calculation of the reward signal.

## Action Space

Collaborators in Tandem Tales act by taking one of four kinds of turns: either
`PASS`ing control to their partner or choosing to `PROPOSE` adding an action to
the story, to commit an action as part of the story with `SUCCEED`, or to
commit an unsuccessful version of an action to the story with `FAIL`.

`PROPOSE` and `SUCCEED` options are semantically congruent and never available
for the same action at the same time. So within TandemTalesEnv, if the world
model has `N` actions, there are `2*N+1` distinct options for the turn that can
be taken at any given step.

As such, the action shape is `(1,)` in the range given by the integer interval
`[0 .. 2*N]`. Within this range, values below `N` represent `PROPOSE`/`SUCCEED`
turns, and values equal to or above `N` represent `FAIL` turns, except for the
value `2*N` which represents the `PASS` turn.

### Action Masking

It is expected that only a small subset of all possible turn choices will be
valid at any given time. Often, especially when responding to a `PROPOSE` turn
or acting in the role of player, this will be a tiny fraction of the possible
choices. For this reason, it is recommended to take advantage of action masking
when using TandemTalesEnv. An appropriate action mask for the current turn is
provided in the info-dict returned by the `reset` and `step` functions
associated with the `'action_mask'` key. This mask can also be accessed with
the `action_masks()` environment method. For compatibility reasons, this mask
is an ndarray with dtype `np.int8`, but a view of this array with dtype
`np.bool` can be accessed from the info-dict with key `b_action_mask`.

## Observation Space

States in Tandem Tales are represented according to a set of variables
specified by the story world. A state is an assignment of one value to every
variable, with the values that a variable can hold also being specified by the
story world. Considering the collaborative game layer of Tandem Tales, it may
also be necessary to distinguish between two types of game mode: either
replying to an offer, or not replying (and so able to propose or pass).

Observations in TandemTalesEnv are a concatenation of one-hot vectors
representing the world state and the game mode. TandemTalesEnv determines the
specific encoding for world variables by collecting all values that are ever
assigned to each variable and organizing the resulting domains in a consistent
fashion (respecting the order the story world lists entities in), ensuring
that the encoding is sufficient, consistent, and concise.

## Starting State

The initial state conditions of episodes in TandemTalesEnv are primarily
determined by the story world used. By default, episodes begin in the initial
world state with the game master as the active role and no offer to reply to.
However, since the environment supports acting in the capacity of either role
but only presents observations and an opportunity to act when appropriate for
the chosen role, it is possible for the actions of a partner agent to alter
the game state prior to the first observation in some situations.

If the environment is in player mode (`as_player=True`), it is possible for the
agent in the game master role to wildly alter the game state before the first
observation. The game will automatically restart internally if choices made
before reset cause it to end, but a clean start can be guaranteed by also
specifying `player_first=True` during environment initialization, in which case
the first turn of the GM role is automatically a `PASS` turn, and the episode
begins in the initial world state with the player as the active role and with
no offer to reply to.

## Termination

An episode in TandemTalesEnv ends if the conditions are met for any one of the
endings described by the story world. By default, it will also end if the
episode length exceeds 1000 steps.

## Rewards

Tandem Tales does not specify any reward signal or close equivalent. Tandem
Tales sessions are collaborative storytelling exercises, and there is no clear
way to broadly define the ideal behaviors or best states of such an experience.

In accordance with this, TandemTalesEnv aims to make the reward signal
customizable, with modular components that can react to a variety of events or
circumstances in the environment and be adapted to specific story worlds.

## Arguments

`world_model` the world model file or WorldModel object of the world to play in
(required).

`render_mode` currently `only` None (no rendering, the default) and `ansi` are
supported (the latter poorly), but `human` will also be enabled.

`allow_invalid` if `False` (the default) invalid actions will raise an exception.

`as_player` if `False` (the default), the game will be played from the GM
perspective. If `True`, it will be played from the player perspective.

`player_first` if `True` the game will behave as if the first turn taken by the
GM role was a `PASS` turn. Defaults to `False`.

`partner_behavior` a custom behavior component that controls the policy
followed by the partner player, and consequently the nondeterminism of
environment transitions.

`reward_handler` a custom reward component that enables fine control over the
rewards given as part of the game.


## License
TandemTalesEnv and related code was developed by Mira Fisher, a PhD student at
the University of Kentucky. This code is owned by the University of Kentucky,
but the author has requested to release it under the GNU General Public License
version 3.0 (GPL 3).