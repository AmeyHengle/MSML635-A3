"""
analysis.py

These are the studies behind the figures I added beyond the ones Part 7 asks for. Each function
here exists to produce evidence for one of the five written questions, rather than to satisfy a
part of the assignment directly. I keep them separate from experiment.py so that the file
holding the required experiment stays exactly as the assignment describes it, and so that a
reader can tell at a glance which numbers are required and which are my own additions.

Everything in this file still uses the assignment's own MDP and its two policies. The studies
that genuinely need a different environment live in extensions.py instead.
"""

import math
import statistics

from environment import generateEpisode
from experiment import TRUE_VALUE
from importance_sampling import (
    computeEffectiveSampleSize,
    computeImportanceRatio,
    computeNaiveAverage,
    computeReturn,
    ordinaryImportanceSampling,
    weightedImportanceSampling,
)

LEFT_LEFT_LABEL = "A-Left, B-Left"
LEFT_RIGHT_LABEL = "A-Left, B-Right"
RIGHT_LABEL = "A-Right"


def makeLogarithmicCheckpoints(numberOfEpisodes, pointsPerDecade=40):
    """
    I build a list of episode counts spaced evenly on a log scale. Almost all of the interesting
    movement in these estimators happens in the first few hundred episodes, so recording the
    running estimate after every single episode would store tens of thousands of points to draw
    a line that is flat over most of its length. Sampling on a log grid keeps the early detail
    and throws away the redundant tail, which matters when I am tracking two hundred runs at once.
    """
    checkpoints = set()
    totalDecades = math.log10(numberOfEpisodes)
    numberOfPoints = max(int(totalDecades * pointsPerDecade), 2)

    for pointIndex in range(numberOfPoints + 1):
        exponent = totalDecades * pointIndex / numberOfPoints
        checkpoints.add(min(int(round(10 ** exponent)), numberOfEpisodes))

    return sorted(checkpoints)


def runNaiveComparison(numberOfEpisodes):
    """
    I track three running estimates side by side: the uncorrected average of the raw returns,
    the ordinary importance sampling estimate and the weighted one. This is my evidence for the
    first written question, because the uncorrected average converges tightly and confidently to
    the wrong answer. The point I want the figure to make is not that the naive estimator is
    noisy but that it is precisely wrong, which is a much harder failure to notice in practice.
    """
    checkpoints = set(makeLogarithmicCheckpoints(numberOfEpisodes))
    history = []

    rawReturnSum = 0.0
    weightedReturnSum = 0.0
    ratioSum = 0.0

    for episodeIndex in range(1, numberOfEpisodes + 1):
        episode = generateEpisode()
        episodeReturn = computeReturn(episode)
        importanceRatio = computeImportanceRatio(episode)

        rawReturnSum += episodeReturn
        weightedReturnSum += importanceRatio * episodeReturn
        ratioSum += importanceRatio

        if episodeIndex in checkpoints:
            history.append((
                episodeIndex,
                rawReturnSum / episodeIndex,
                weightedReturnSum / episodeIndex,
                weightedReturnSum / ratioSum if ratioSum > 0 else 0.0,
            ))

    return history


def measureTrajectoryFrequencies(numberOfEpisodes):
    """
    I count how often each of the three possible trajectories actually occurs. The MDP only
    admits three, so comparing the measured frequencies against the 0.5, 0.25 and 0.25 that the
    behavior policy implies is a direct check that my generator matches the environment I was
    given. I assert this in the test suite as well, but having it as a figure means the reader
    can see the generator is sound before being asked to trust anything built on top of it.
    """
    counts = {RIGHT_LABEL: 0, LEFT_RIGHT_LABEL: 0, LEFT_LEFT_LABEL: 0}

    for _ in range(numberOfEpisodes):
        episode = generateEpisode()
        actions = tuple(action for _, action, _ in episode)

        if actions == ("Right",):
            counts[RIGHT_LABEL] += 1
        elif actions == ("Left", "Right"):
            counts[LEFT_RIGHT_LABEL] += 1
        else:
            counts[LEFT_LEFT_LABEL] += 1

    return {label: count / numberOfEpisodes for label, count in counts.items()}


def collectImportanceRatios(numberOfEpisodes):
    """
    I collect the importance ratio of every episode in one batch, together with the effective
    sample size those ratios imply. The histogram of these values is the clearest way I can show
    that three quarters of my data is discarded outright, and the effective sample size turns
    that observation into a single number I can quote in my answer to the second question.
    """
    importanceRatios = [computeImportanceRatio(generateEpisode()) for _ in range(numberOfEpisodes)]
    effectiveSampleSize = computeEffectiveSampleSize(importanceRatios)
    return importanceRatios, effectiveSampleSize


def runVarianceScalingStudy(episodeCounts, repetitionCounts):
    """
    I measure how the spread of each estimator shrinks as the number of episodes grows, and
    compare it against the square root of 12 over N that theory predicts for the ordinary
    estimator. This is the strongest validation I have: matching theory at a single sample size
    could be luck, but matching the slope across several orders of magnitude means the estimator
    has the right convergence rate. I also record the root mean squared error of both estimators,
    which folds bias and variance into the one quantity that actually matters in practice. I take
    fewer repetitions at the larger episode counts because the cost grows with their product and
    the spread is already well determined there.
    """
    results = []

    for numberOfEpisodes, repetitions in zip(episodeCounts, repetitionCounts):
        ordinaryEstimates = []
        weightedEstimates = []

        for _ in range(repetitions):
            returnsAndRatios = [
                (computeReturn(episode), computeImportanceRatio(episode))
                for episode in (generateEpisode() for _ in range(numberOfEpisodes))
            ]
            episodeReturns = [pair[0] for pair in returnsAndRatios]
            importanceRatios = [pair[1] for pair in returnsAndRatios]

            ordinaryEstimates.append(ordinaryImportanceSampling(importanceRatios, episodeReturns))
            weightedEstimates.append(weightedImportanceSampling(importanceRatios, episodeReturns))

        results.append({
            "episodes": numberOfEpisodes,
            "ordinaryStd": statistics.stdev(ordinaryEstimates),
            "weightedStd": statistics.stdev(weightedEstimates),
            "predictedStd": (12.0 / numberOfEpisodes) ** 0.5,
            "ordinaryRmse": computeRootMeanSquaredError(ordinaryEstimates),
            "weightedRmse": computeRootMeanSquaredError(weightedEstimates),
        })

    return results


def computeRootMeanSquaredError(estimates):
    """
    I compute the root mean squared error of a list of estimates against the known true value of
    2. I use this rather than the standard deviation when I want a single number that penalises
    bias and variance together, which is the comparison that actually decides which estimator I
    would rather use on a finite sample.
    """
    squaredErrors = [(estimate - TRUE_VALUE) ** 2 for estimate in estimates]
    return (sum(squaredErrors) / len(squaredErrors)) ** 0.5


def runBiasStudy(episodeCounts, repetitions):
    """
    I measure the average estimate at small sample sizes, where the bias of the weighted
    estimator is actually visible. With a single episode only a quarter of runs contain any
    trajectory consistent with the target policy, so the weighted estimator is undefined in the
    rest and averages far below the true value. As the episode count grows that case becomes
    rare and the estimator climbs towards 2, which is consistency in action. The ordinary
    estimator sits on 2 at every sample size, including N equal to one, because it is unbiased.
    I report the standard error of each mean so the figure can show that the gap at small N is
    real rather than noise.
    """
    results = []

    for numberOfEpisodes in episodeCounts:
        ordinaryEstimates = []
        weightedEstimates = []

        for _ in range(repetitions):
            returnsAndRatios = [
                (computeReturn(episode), computeImportanceRatio(episode))
                for episode in (generateEpisode() for _ in range(numberOfEpisodes))
            ]
            episodeReturns = [pair[0] for pair in returnsAndRatios]
            importanceRatios = [pair[1] for pair in returnsAndRatios]

            ordinaryEstimates.append(ordinaryImportanceSampling(importanceRatios, episodeReturns))
            weightedEstimates.append(weightedImportanceSampling(importanceRatios, episodeReturns))

        results.append({
            "episodes": numberOfEpisodes,
            "ordinaryMean": statistics.mean(ordinaryEstimates),
            "weightedMean": statistics.mean(weightedEstimates),
            "ordinaryStandardError": statistics.stdev(ordinaryEstimates) / math.sqrt(repetitions),
            "weightedStandardError": statistics.stdev(weightedEstimates) / math.sqrt(repetitions),
        })

    return results


def runConvergenceEnvelope(numberOfRuns, numberOfEpisodes):
    """
    I repeat the convergence tracking across many independent runs and keep the whole spread at
    each checkpoint rather than a single trace. My original convergence figure used one run,
    which leaves it open to the objection that I picked a favourable seed. Plotting the band
    between the fifth and ninety fifth percentile across two hundred runs answers that objection
    directly, and it shows the ordinary estimator's funnel narrowing slowly while the weighted
    estimator collapses onto the true value almost immediately.
    """
    checkpoints = makeLogarithmicCheckpoints(numberOfEpisodes)
    checkpointSet = set(checkpoints)

    ordinaryTraces = [[] for _ in checkpoints]
    weightedTraces = [[] for _ in checkpoints]

    for _ in range(numberOfRuns):
        weightedReturnSum = 0.0
        ratioSum = 0.0
        checkpointIndex = 0

        for episodeIndex in range(1, numberOfEpisodes + 1):
            episode = generateEpisode()
            episodeReturn = computeReturn(episode)
            importanceRatio = computeImportanceRatio(episode)

            weightedReturnSum += importanceRatio * episodeReturn
            ratioSum += importanceRatio

            if episodeIndex in checkpointSet:
                ordinaryTraces[checkpointIndex].append(weightedReturnSum / episodeIndex)
                weightedTraces[checkpointIndex].append(
                    weightedReturnSum / ratioSum if ratioSum > 0 else 0.0
                )
                checkpointIndex += 1

    return checkpoints, summarisePercentiles(ordinaryTraces), summarisePercentiles(weightedTraces)


def summarisePercentiles(traces, lowerPercentile=5, upperPercentile=95):
    """
    I reduce each checkpoint's list of estimates to a median and a pair of percentiles. I chose
    percentiles rather than a multiple of the standard deviation because the distribution of the
    ordinary estimator is visibly skewed at small sample sizes, and a symmetric band drawn around
    the mean would misrepresent it.
    """
    summary = []

    for values in traces:
        orderedValues = sorted(values)
        summary.append((
            percentileOf(orderedValues, lowerPercentile),
            percentileOf(orderedValues, 50),
            percentileOf(orderedValues, upperPercentile),
        ))

    return summary


def percentileOf(orderedValues, percentile):
    """
    I pick a percentile out of an already sorted list by linear interpolation between the two
    neighbouring entries. I wrote it out rather than reaching for numpy because matplotlib is the
    only dependency this project has, and I would rather not add another one for four lines.
    """
    if not orderedValues:
        return 0.0

    position = (len(orderedValues) - 1) * percentile / 100.0
    lowerIndex = int(math.floor(position))
    upperIndex = min(lowerIndex + 1, len(orderedValues) - 1)
    interpolationWeight = position - lowerIndex

    return (
        orderedValues[lowerIndex] * (1 - interpolationWeight)
        + orderedValues[upperIndex] * interpolationWeight
    )


def runSamplingDistributionStudy(numberOfExperiments, numberOfEpisodes):
    """
    I collect the ordinary estimate from many independent experiments so that I can look at the
    shape of its sampling distribution, not just its first two moments. If my implementation is
    correct the histogram should look normal and should match the density predicted by the
    central limit theorem with a standard deviation of the square root of 12 over N. Matching a
    whole distribution is a far stronger claim than matching a mean and a standard deviation.
    """
    estimates = []

    for _ in range(numberOfExperiments):
        returnsAndRatios = [
            (computeReturn(episode), computeImportanceRatio(episode))
            for episode in (generateEpisode() for _ in range(numberOfEpisodes))
        ]
        episodeReturns = [pair[0] for pair in returnsAndRatios]
        importanceRatios = [pair[1] for pair in returnsAndRatios]
        estimates.append(ordinaryImportanceSampling(importanceRatios, episodeReturns))

    return estimates


def computeTargetTrajectoryProbabilities():
    """
    I return the probability of each trajectory under both policies so that I can draw them side
    by side. The behavior policy spreads its mass across all three trajectories while the target
    policy puts everything on one, and that mismatch is the whole reason a correction is needed.
    These are exact numbers from the policy definitions rather than measurements.
    """
    return {
        RIGHT_LABEL: (0.5, 0.0),
        LEFT_RIGHT_LABEL: (0.25, 0.0),
        LEFT_LEFT_LABEL: (0.25, 1.0),
    }
