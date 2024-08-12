The general idea is to separate logic into three class families:

* Drivers: provide valuations for the automaton's atomic propositions (AP)
* Runners: take values from drivers and evolve the automaton accordingly
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

# Hooks

Hooks allow to customize the behaviour of `hoa-exec` by firing a *reaction* when
some specific *condition* is met. These are still far from being laid out in
detail nor implemented.

The idea for now is to provide two extension point at every iteration of the
runner: *before transition* (but after available transitions are computed), and
*after transition*, i.e., when the automaton moves to the successor state.

Before-transition conditions might include

* Deadlock: no feasible successor state is found
* Nondeterminism: multiple successor states are available

* Stutter: the automaton takes a self-transition (from current state unto
  itself)
* State change: the automaton moves from the current state to another one
* Transition: either state change or stutter
* Acceptance: an acceptance condition is met

And reactions might include:

* Log: write down information to a log file, or to stdout
* Reset: resets the automaton to its initial state
* Goto: set the automaton's current state (this may be a family of reactions 
  with different policies for choosing the state)
* Exit: quit `hoa-exec`
* Composite: a composition of two or more of the above.

Some examples why hooks are useful:

* Setting up `hoa-exec` for runtime verification (RV) would entail
  adding a hook whereby meeting the acceptance condition (i.e., detecting a
  violation) should trigger a Log action and possibly a Reset/Goto.

* By using a Nondeterminism + Goto hook the user may customize how the
  automaton will resolve nondeterminism.