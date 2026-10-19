"""
extensions.py

Everything in this file goes beyond the environment the assignment specifies, and I have kept
it in a separate module for exactly that reason: nothing here touches the numbers I report for
Parts 1 to 7.

My reason for adding it is that two of the written questions cannot honestly be answered from
the given MDP. Question 2 asks why the importance ratio can become very large, but in this
environment it is capped at 4, because episodes are at most two steps long and the behavior
policy never assigns a probability below one half. Question 4 asks which estimator is unbiased,
but the given target policy is deterministic, which makes the weighted estimator exactly correct
here and hides the bias it has in general. To show either effect I have to vary something, so I
vary one thing at a time and say plainly which figure came from which variation.

Each study below has an analytical answer I derived first, so the simulation is a check on my
reasoning rather than a substitute for it.
"""

import random
import statistics

from environment import generateEpisode, makePolicy
from importance_sampling import (
    computeImportanceRatio,
    computeReturn,
    ordinaryImportanceSampling,
    weightedImportanceSampling,
)

CHAIN_FINAL_REWARD = 2


def generateChainEpisode(chainLength):
    """
    I generate an episode in a chain of the given length, which is the assignment's MDP stretched
    out so that episodes can be longer than two steps. In each state Left continues along the
    chain and Right terminates immediately with no reward, and reaching the end of the chain with
    Left pays out 2. The behavior policy is a fair coin at every state and the target policy
    always goes Left, exactly as in the assignment, so the only thing I have changed is how many
    decisions an episode contains. At a chain length of two this is the original MDP as far as the
    importance ratio is concerned, which is what lets me check the extension against known results.
    """
    episode = []

    for stateIndex in range(chainLength):
        action = "Left" if random.random() < 0.5 else "Right"

        if action == "Right":
            episode.append((stateIndex, action, 0))
            return episode

        reward = CHAIN_FINAL_REWARD if stateIndex == chainLength - 1 else 0
        episode.append((stateIndex, action, reward))

    return episode


def computeChainRatio(episode, chainLength):
    """
    I compute the importance ratio for a chain episode. The target policy always goes Left, so any
    episode containing a Right action has a ratio of zero, and an episode that walked the whole
    chain picks up a factor of two at every one of its steps. The ratio is therefore either zero
    or two to the power of the chain length, which is the exponential growth that question two is
    asking about, made explicit.
    """
    for _, action, _ in episode:
        if action == "Right":
            return 0.0
    return 2.0 ** chainLength


def computeChainTheory(chainLength):
    """
    I derive the three quantities this study is about before measuring any of them. A surviving
    episode has a ratio of two to the chain length and occurs with the reciprocal of that
    probability, so the estimator remains unbiased at 2. Its variance is four times two to the
    chain length, minus four, which doubles with every extra step. At a chain length of two this
    gives twelve, which is exactly the variance I derived for the assignment's own MDP, and that
    agreement is my check that the extension is set up correctly.
    """
    survivingRatio = 2.0 ** chainLength
    survivalProbability = 1.0 / survivingRatio
    varianceOfWeightedReturn = survivalProbability * (survivingRatio * CHAIN_FINAL_REWARD) ** 2 - 4.0

    return {
        "chainLength": chainLength,
        "survivingRatio": survivingRatio,
        "survivalProbability": survivalProbability,
        "variance": varianceOfWeightedReturn,
    }


def runChainLengthStudy(chainLengths, numberOfEpisodes):
    """
    I measure the realised ratio, the survival rate and the variance of the weighted return for
    each chain length, so that I can plot the measurements on top of the analytical curves. I keep
    the chain lengths modest because the survival probability halves with each extra step, and
    beyond about ten steps a sample of this size contains too few surviving episodes to estimate a
    variance from. That limitation is itself part of the answer to question two, so I would rather
    show where the measurement breaks down than quietly stop before it does.
    """
    results = []

    for chainLength in chainLengths:
        weightedReturns = []
        survivorCount = 0

        for _ in range(numberOfEpisodes):
            episode = generateChainEpisode(chainLength)
            episodeReturn = computeReturn(episode)
            importanceRatio = computeChainRatio(episode, chainLength)

            if importanceRatio > 0:
                survivorCount += 1
            weightedReturns.append(importanceRatio * episodeReturn)

        theory = computeChainTheory(chainLength)
        theory["measuredSurvivalRate"] = survivorCount / numberOfEpisodes
        theory["measuredVariance"] = statistics.pvariance(weightedReturns)
        results.append(theory)

    return results


def runBehaviorProbabilitySweep(leftProbabilities, numberOfEpisodes):
    """
    I hold the assignment's MDP and target policy fixed and vary only how often the behavior
    policy takes the action the target policy wants. This is the second mechanism that inflates
    the ratio: the ratio for the surviving trajectory is the reciprocal of the behavior
    probability squared, so a behavior policy that rarely chooses Left produces enormous weights
    on the few episodes that do. I think this is the more realistic of the two mechanisms, because
    in practice the logged policy is usually fixed and the mismatch is what varies.
    """
    results = []

    for leftProbability in leftProbabilities:
        behaviorPolicy = makePolicy(leftProbability)
        importanceRatios = []
        episodeReturns = []

        for _ in range(numberOfEpisodes):
            episode = generateEpisode(behaviorPolicy)
            episodeReturns.append(computeReturn(episode))
            importanceRatios.append(computeImportanceRatio(episode, behaviorPolicy=behaviorPolicy))

        weightedReturns = [
            ratio * episodeReturn
            for ratio, episodeReturn in zip(importanceRatios, episodeReturns)
        ]

        results.append({
            "leftProbability": leftProbability,
            "theoreticalRatio": (1.0 / leftProbability) ** 2,
            "measuredMaximumRatio": max(importanceRatios),
            "survivalRate": sum(1 for ratio in importanceRatios if ratio > 0) / numberOfEpisodes,
            "variance": statistics.pvariance(weightedReturns),
        })

    return results


def computeSoftTargetValue(leftProbability):
    """
    I derive the true value of the start state when the target policy chooses Left with the given
    probability rather than always. Working backwards, the value of B is twice that probability,
    and the value of A is that probability times the value of B plus the chance of going Right and
    collecting 1. I need this exact number because the whole point of the study is to measure a
    bias, and a measured bias is meaningless without a true value to measure it against.
    """
    valueOfB = 2.0 * leftProbability
    return leftProbability * valueOfB + (1.0 - leftProbability) * 1.0


def runSoftTargetBiasStudy(leftProbabilities, numberOfEpisodes, repetitions):
    """
    I soften the target policy so that more than one trajectory survives the ratio, which is the
    condition under which the weighted estimator's bias actually appears. With the assignment's
    deterministic target the surviving episodes all carry the same ratio and the same return, so
    the weighted estimator is exactly right and its bias is invisible. Once several distinct
    trajectories survive with different weights, the estimator becomes a genuine ratio of two
    random variables and it no longer sits on the true value. I report both estimators against the
    exact value for each softened policy so the difference between unbiased and merely consistent
    is visible rather than asserted.
    """
    results = []

    for leftProbability in leftProbabilities:
        targetPolicy = makePolicy(leftProbability)
        trueValue = computeSoftTargetValue(leftProbability)

        ordinaryEstimates = []
        weightedEstimates = []

        for _ in range(repetitions):
            importanceRatios = []
            episodeReturns = []

            for _ in range(numberOfEpisodes):
                episode = generateEpisode()
                episodeReturns.append(computeReturn(episode))
                importanceRatios.append(computeImportanceRatio(episode, targetPolicy=targetPolicy))

            ordinaryEstimates.append(ordinaryImportanceSampling(importanceRatios, episodeReturns))
            weightedEstimates.append(weightedImportanceSampling(importanceRatios, episodeReturns))

        results.append({
            "leftProbability": leftProbability,
            "trueValue": trueValue,
            "ordinaryMean": statistics.mean(ordinaryEstimates),
            "weightedMean": statistics.mean(weightedEstimates),
            "ordinaryBias": statistics.mean(ordinaryEstimates) - trueValue,
            "weightedBias": statistics.mean(weightedEstimates) - trueValue,
            "ordinaryStandardError": statistics.stdev(ordinaryEstimates) / repetitions ** 0.5,
            "weightedStandardError": statistics.stdev(weightedEstimates) / repetitions ** 0.5,
        })

    return results
