"""
environment.py

I keep the MDP itself in this file and nothing else. The assignment asks for off-policy
prediction, so the environment has to stay completely unaware of which policy is being
evaluated. Any file that needs an episode imports generateEpisode from here. I started
from the starter code the instructor provided and kept the transition logic identical,
changing only the naming so it follows the camelCase convention I use across the repo.
"""

import random

STATES = ["A", "B"]
TERMINAL_STATE = "Terminal"
ACTIONS = ["Left", "Right"]

BEHAVIOR_POLICY = {
    "A": {"Left": 0.5, "Right": 0.5},
    "B": {"Left": 0.5, "Right": 0.5},
}

TARGET_POLICY = {
    "A": {"Left": 1.0, "Right": 0.0},
    "B": {"Left": 1.0, "Right": 0.0},
}


def selectAction(state, policy):
    """
    I sample a single action from whichever policy I am handed. I pull the action names and
    their probabilities out of the dictionary in the same order so that random.choices lines
    the weights up with the correct actions. I kept this as its own function because the
    behavior policy is what generates the data while the target policy is only ever used for
    the ratio, and separating the sampling step makes that distinction explicit.
    """
    actionProbabilities = policy[state]
    return random.choices(
        list(actionProbabilities.keys()),
        weights=list(actionProbabilities.values()),
    )[0]


def applyTransition(state, action):
    """
    I encode the transition table from the assignment here. From A, Left moves me to B with
    reward 0 and Right terminates with reward 1. From B, Left terminates with reward 2 and
    Right terminates with reward 0. I return the next state and the reward together because
    every caller needs both, and keeping them in one function means the transition rules live
    in exactly one place rather than being repeated inside the episode loop.
    """
    if state == "A":
        if action == "Left":
            return "B", 0
        return TERMINAL_STATE, 1

    if action == "Left":
        return TERMINAL_STATE, 2
    return TERMINAL_STATE, 0


def makePolicy(leftProbability):
    """
    I build a policy that chooses Left with the given probability in both states. The assignment
    only needs the two fixed policies defined above, but several of my extra figures ask what
    happens as a policy is varied, and generating those policies here keeps the rest of the code
    from having to know how a policy dictionary is shaped.
    """
    rightProbability = 1.0 - leftProbability
    return {
        "A": {"Left": leftProbability, "Right": rightProbability},
        "B": {"Left": leftProbability, "Right": rightProbability},
    }


def generateEpisode(behaviorPolicy=BEHAVIOR_POLICY):
    """
    I generate one complete episode by following the behavior policy from the start state A
    until I reach the terminal state. Every episode begins at A, which is what makes this
    experiment an estimate of V(A) specifically. I record each step as a (state, action,
    reward) tuple so that later on I can walk the same trajectory twice, once to accumulate
    the return and once to accumulate the importance sampling ratio. The policy is an argument
    with the assignment's policy as its default, so every result I report uses exactly the
    behavior policy I was given while my sweeps can still vary it without a second generator.
    """
    episode = []
    currentState = "A"

    while currentState != TERMINAL_STATE:
        selectedAction = selectAction(currentState, behaviorPolicy)
        nextState, reward = applyTransition(currentState, selectedAction)
        episode.append((currentState, selectedAction, reward))
        currentState = nextState

    return episode


def setRandomSeed(seedValue):
    """
    I expose the seed through the environment so that main.py and test_cases.py can both make
    a run reproducible without importing random themselves. I need reproducible runs because
    the whole point of this assignment is comparing the spread of two estimators, and I want
    to be able to regenerate the exact numbers I report.
    """
    random.seed(seedValue)
