"""
experiment.py

This file runs the experiment described in Parts 1, 2, 3, 4, 5 and 6 of the assignment. One
experiment means ten thousand episodes turned into a single pair of estimates, and the full
study repeats that twenty times so that I can measure how much each estimator moves from one
repetition to the next. The statistics I report at the end are over those twenty values, not
over individual episodes, because the question I am answering is how reliable each estimator
is as a procedure.
"""

import statistics

from environment import generateEpisode
from importance_sampling import (
    computeImportanceRatio,
    computeReturn,
    ordinaryImportanceSampling,
    weightedImportanceSampling,
)

TRUE_VALUE = 2.0
DEFAULT_EPISODE_COUNT = 10000
DEFAULT_EXPERIMENT_COUNT = 20


def runSingleExperiment(numberOfEpisodes=DEFAULT_EPISODE_COUNT, trackHistory=False):
    """
    I run one experiment: generate the requested number of episodes, and for each one compute
    its return and its importance ratio before collecting both estimates at the end. I walk
    each episode only once and store the two numbers, because the episode itself is not needed
    afterwards and holding ten thousand trajectories in memory would be wasteful. When
    trackHistory is on I also record the running estimate after every episode so that I can
    plot how each estimator converges, which is the evidence I use for the last written
    question. I keep running sums rather than recomputing from scratch so the history costs
    almost nothing.
    """
    importanceRatios = []
    episodeReturns = []
    history = []

    runningWeightedSum = 0.0
    runningRatioSum = 0.0

    for episodeIndex in range(1, numberOfEpisodes + 1):
        episode = generateEpisode()
        episodeReturn = computeReturn(episode)
        importanceRatio = computeImportanceRatio(episode)

        episodeReturns.append(episodeReturn)
        importanceRatios.append(importanceRatio)

        if trackHistory:
            runningWeightedSum += importanceRatio * episodeReturn
            runningRatioSum += importanceRatio
            runningOrdinary = runningWeightedSum / episodeIndex
            runningWeighted = (
                runningWeightedSum / runningRatioSum if runningRatioSum > 0 else 0.0
            )
            history.append((episodeIndex, runningOrdinary, runningWeighted))

    ordinaryEstimate = ordinaryImportanceSampling(importanceRatios, episodeReturns)
    weightedEstimate = weightedImportanceSampling(importanceRatios, episodeReturns)

    return ordinaryEstimate, weightedEstimate, history


def runAllExperiments(
    numberOfExperiments=DEFAULT_EXPERIMENT_COUNT,
    numberOfEpisodes=DEFAULT_EPISODE_COUNT,
    verbose=True,
):
    """
    I repeat the experiment the requested number of times and keep both estimates from each
    repetition. I deliberately do not reseed between repetitions, because each repetition is
    supposed to be an independent draw of the whole procedure and reseeding would make them
    identical. I print each repetition as it finishes so that a long run shows progress rather
    than sitting silent, and so the spread between the two columns is visible immediately.
    """
    ordinaryEstimates = []
    weightedEstimates = []

    if verbose:
        print(f"{'Run':>4}  {'Ordinary IS':>14}  {'Weighted IS':>14}")
        print("-" * 36)

    for experimentIndex in range(1, numberOfExperiments + 1):
        ordinaryEstimate, weightedEstimate, _ = runSingleExperiment(numberOfEpisodes)
        ordinaryEstimates.append(ordinaryEstimate)
        weightedEstimates.append(weightedEstimate)

        if verbose:
            print(f"{experimentIndex:>4}  {ordinaryEstimate:>14.6f}  {weightedEstimate:>14.6f}")

    return ordinaryEstimates, weightedEstimates


def computeStatistics(estimates):
    """
    I summarise a list of estimates with its mean and sample standard deviation. I use the
    sample version with the n-1 correction because my twenty repetitions are a sample of the
    estimator's behaviour rather than the entire population of possible runs. The mean tells
    me whether the estimator is centred on the true value of 2, and the standard deviation is
    the quantity the assignment actually wants me to compare between the two estimators.
    """
    meanValue = statistics.mean(estimates)
    standardDeviation = statistics.stdev(estimates) if len(estimates) > 1 else 0.0
    return meanValue, standardDeviation


def computeBehaviorPolicyValue():
    """
    I compute the value of the start state under the behavior policy analytically. This is not
    one of the required parts, but it is the clearest way I can show why importance sampling is
    necessary at all: if I ignored the policy mismatch and simply averaged the raw returns I
    would converge to this number instead of the true target value. The three terms are the
    three possible trajectories weighted by how often the behavior policy produces them.
    """
    return 0.5 * 1 + 0.25 * 0 + 0.25 * 2


def computeTheoreticalOrdinaryVariance():
    """
    I derive the variance of a single weighted return under the behavior policy so that I can
    check my simulation against theory rather than against intuition. Only the Left-Left
    trajectory has a non-zero product, and it occurs a quarter of the time with rho = 4 and
    G = 2, so the second moment is 0.25 * (4 * 2)^2 = 16 and the variance is 16 - 2^2 = 12.
    Dividing by the number of episodes gives the variance I should expect my ordinary estimate
    to show across repetitions.
    """
    secondMoment = 0.25 * ((4.0 * 2.0) ** 2)
    return secondMoment - TRUE_VALUE ** 2
