The general idea is to separate logic into three class families:

* Drivers: provide valuations for the automaton's atomic propositions (AP)
* Runners: take values from drivers and evolve the automata accordingly
* Hooks: perform additional actions when triggered

# Drivers

Drivers simply read in, or generate, a valuation for APs.

At the moment we support a (possibly biased) random driver and an interactive
one. The plan is to add at least support for JSON inputs (either dictionaries
or arrays).

Drivers compose, i.e., the user may configure `hoa-exec` so that different 
APs get values from different sources.

# Runners

The main focus at the moment is on implementing a discrete runner which simply
does the following:

1. Take inputs from drivers
2. Compute the next state (possibly invoking hooks)
3. Invoke post-transition hooks
4. Repeat

We are currently implementing a _synchronous_ semantics: the driver
gives values to every AP at every iteration.
_Asynchronous_ semantics may be desirable sometimes, where the driver provides
partial updates and all other APs are assumed to stay unchanged. (WIP)

By default, if no transition is available, the automaton will remain in its current state (stutter). 


We also support running multiple automata at the same time. At the moment
these all receive the same inputs.

# Actions

Actions allow to configure a runner's behaviour when specific situations arise.
At the moment we implement two extension points:

* When no successor to the current state is available to the given input (deadlock);
* When multiple transitions are available from the current state under the given
  input (nondeterminism).

Actions include:

* `Log`: write down information to a log file, or to stdout
* `Reset`: resets the automaton to its initial state
* `Quit`: quit `hoa-exec`
* `Composite`: a composition of two or more of the above.

For instance, the user may want to log a message and reset the automaton on deadlock.
This can be achieved by adding a `Log()` and a `Reset()` actions to the runner's
deadlock actions. (Only via API currently; scripting support is planned)

# Hooks

Hooks allow to further customize the behaviour of `hoa-exec` by firing an
action when some specific *condition* is met. These are still far from being
laid out in detail nor implemented.

At the moment, hooks are tested and fired *after* a transition has taken place.

Before-transition conditions might include

* Always: fires after every transition
* Bound: a certain number of step has been performed
* Reach: a specific state has been reached
* Acceptance: an acceptance condition is met

Some examples why hooks are useful:

* We might want print a message when specific states have been reached

* Setting up `hoa-exec` for runtime verification (RV) would entail
  adding a hook whereby meeting the acceptance condition (i.e., detecting a
  violation or deadlock) should trigger a Log action and possibly a Reset.
