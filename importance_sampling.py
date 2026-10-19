"""
importance_sampling.py

This file holds the maths of the assignment: the return of an episode, the importance
sampling ratio for that episode, and the two estimators built from those quantities. I kept
all four functions free of any randomness so that they are pure functions of an episode or
of a list of numbers, which is what lets me test them against values I worked out by hand.
"""

from environment import BEHAVIOR_POLICY, TARGET_POLICY


def computeReturn(episode, discountFactor=1.0):
    """
    I compute the total return G for one episode. The assignment does not specify a discount,
    so my default is gamma = 1 and G reduces to the plain sum of the rewards. I still wrote
    the discounted form because it costs nothing and it documents the assumption instead of
    hiding it, and it lets me demonstrate that gamma = 1 is a deliberate choice rather than an
    oversight. The exponent is the time index of the step, so the first reward is undiscounted.
    """
    totalReturn = 0.0
    for timeStep, (_, _, reward) in enumerate(episode):
        totalReturn += (discountFactor ** timeStep) * reward
    return totalReturn


def computeImportanceRatio(episode, targetPolicy=TARGET_POLICY, behaviorPolicy=BEHAVIOR_POLICY):
    """
    I compute the importance sampling ratio rho as the product of pi(a|s) / b(a|s) over every
    step of the episode. Because my target policy is deterministic, any episode containing a
    Right action gets a numerator of zero and the whole product collapses to zero, so I break
    out of the loop as soon as that happens. The early exit is only an efficiency detail since
    a product that has hit zero can never recover, but it also makes the behaviour obvious to
    anyone reading the function. The episodes here start at A, so this full-trajectory ratio
    is the correct weight for estimating V(A). Both policies default to the ones the assignment
    specifies, so my reported results always use them, while the sweeps I added for the written
    questions can pass in a modified policy instead of reimplementing the ratio.
    """
    importanceRatio = 1.0

    for state, action, _ in episode:
        targetProbability = targetPolicy[state][action]
        behaviorProbability = behaviorPolicy[state][action]

        if targetProbability == 0.0:
            return 0.0

        importanceRatio *= targetProbability / behaviorProbability

    return importanceRatio


def ordinaryImportanceSampling(importanceRatios, episodeReturns):
    """
    I implement the ordinary estimator, which averages the weighted returns over N, where N is
    every episode I generated and not just the ones that survived the ratio. Dividing by the
    full count is exactly what makes this estimator unbiased: the episodes that the target
    policy would never produce contribute a genuine zero to the average rather than being
    quietly dropped. In this MDP three quarters of my episodes contribute zero, which is the
    reason the estimator is noisier than the weighted one despite being unbiased.
    """
    numberOfEpisodes = len(episodeReturns)
    if numberOfEpisodes == 0:
        return 0.0

    weightedSum = sum(
        ratio * episodeReturn
        for ratio, episodeReturn in zip(importanceRatios, episodeReturns)
    )
    return weightedSum / numberOfEpisodes


def weightedImportanceSampling(importanceRatios, episodeReturns):
    """
    I implement the weighted estimator, which divides by the sum of the ratios instead of by N.
    That turns the estimate into a weighted average of the returns I actually observed, so it
    can never fall outside the range of those returns no matter how extreme a single ratio
    becomes. I guard against a zero denominator because the estimator is undefined when no
    episode is consistent with the target policy. With ten thousand episodes that case is
    effectively impossible here, but the estimator should still be well defined for any input.
    """
    ratioSum = sum(importanceRatios)
    if ratioSum == 0.0:
        return 0.0

    weightedSum = sum(
        ratio * episodeReturn
        for ratio, episodeReturn in zip(importanceRatios, episodeReturns)
    )
    return weightedSum / ratioSum


def computeNaiveAverage(episodeReturns):
    """
    I average the returns with no importance correction at all. This is not one of the required
    estimators, it is the mistake the assignment is warning against, and I compute it so that I
    can show it converging to the value of the behavior policy rather than the target policy.
    Having it as a function lets me plot it beside the two real estimators on the same axes.
    """
    if not episodeReturns:
        return 0.0
    return sum(episodeReturns) / len(episodeReturns)


def computeEffectiveSampleSize(importanceRatios):
    """
    I compute the effective sample size of a set of weights, which is the squared sum of the
    ratios divided by the sum of their squares. It answers the question of how many equally
    weighted episodes my weighted episodes are really worth. In this MDP three quarters of the
    ratios are zero and the rest are all equal, so the answer comes out at a quarter of the
    episode count, which is a far more honest description of my sample than the raw count of
    ten thousand. I return 0 when every ratio is zero, since no episode carries any information.
    """
    ratioSum = sum(importanceRatios)
    squaredSum = sum(ratio * ratio for ratio in importanceRatios)
    if squaredSum == 0.0:
        return 0.0
    return (ratioSum ** 2) / squaredSum
