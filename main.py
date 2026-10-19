"""
main.py

This is the entry point that runs the whole assignment end to end: it generates the episodes,
computes both estimators across twenty repetitions, prints the statistics and saves the
figures. I put the command line options here so that the marker can reproduce my numbers with
a single command, and so that I can shrink the episode count while developing without editing
any source.

    python main.py                      the full assignment, 20 experiments of 10000 episodes
    python main.py --seed 7             a different reproducible run
    python main.py --episodes 1000      a quicker run with fewer episodes per experiment
    python main.py --no-plots           statistics only, no figures written
"""

import argparse
import os

from environment import setRandomSeed
from experiment import (
    DEFAULT_EPISODE_COUNT,
    DEFAULT_EXPERIMENT_COUNT,
    TRUE_VALUE,
    computeBehaviorPolicyValue,
    computeStatistics,
    computeTheoreticalOrdinaryVariance,
    runAllExperiments,
    runSingleExperiment,
)
from analysis import (
    collectImportanceRatios,
    computeTargetTrajectoryProbabilities,
    measureTrajectoryFrequencies,
    runBiasStudy,
    runConvergenceEnvelope,
    runNaiveComparison,
    runSamplingDistributionStudy,
    runVarianceScalingStudy,
)
from extensions import runBehaviorProbabilitySweep, runChainLengthStudy, runSoftTargetBiasStudy
from plots import (
    plotBiasVersusEpisodes,
    plotConvergence,
    plotConvergenceEnvelope,
    plotEstimateSpread,
    plotEstimatesPerExperiment,
    plotNaiveVersusCorrected,
    plotRatioDistribution,
    plotRatioExplosion,
    plotSamplingDistribution,
    plotSoftTargetBias,
    plotTrajectoryDistributions,
    plotVarianceScaling,
)

FIGURES_DIRECTORY = "figures"

SCALING_EPISODE_COUNTS = [10, 100, 1000, 10000, 100000]
SCALING_REPETITIONS = [400, 300, 200, 100, 40]
BIAS_EPISODE_COUNTS = [1, 2, 3, 5, 10, 20, 50, 100]
CHAIN_LENGTHS = list(range(1, 11))
BEHAVIOR_LEFT_PROBABILITIES = [0.9, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05]
SOFT_TARGET_LEFT_PROBABILITIES = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
SOFT_TARGET_EPISODE_COUNT = 10


def parseArguments():
    """
    I read the command line options. The defaults are exactly the values the assignment
    specifies, so running the file with no arguments reproduces the required experiment, and
    every flag is only there to make the run faster or differently seeded while I develop.
    """
    parser = argparse.ArgumentParser(
        description="Off-policy Monte Carlo prediction using importance sampling."
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODE_COUNT,
                        help="episodes generated per experiment (default 10000)")
    parser.add_argument("--experiments", type=int, default=DEFAULT_EXPERIMENT_COUNT,
                        help="number of repetitions of the experiment (default 20)")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed so a run can be reproduced (default 0)")
    parser.add_argument("--no-plots", dest="noPlots", action="store_true",
                        help="print the statistics but do not write any figures")
    parser.add_argument("--quick", action="store_true",
                        help="write only the figures the assignment requires, skipping the extras")
    parser.add_argument("--figures-dir", dest="figuresDirectory", default=FIGURES_DIRECTORY,
                        help="directory the figures are written to (default figures)")
    return parser.parse_args()


def printSectionHeading(headingText):
    """
    I print a consistent heading before each block of output. The console output is part of
    what I submit, so I want it readable rather than a wall of numbers with no structure.
    """
    print()
    print(headingText)
    print("=" * len(headingText))


def printWhyImportanceSamplingIsNeeded():
    """
    I print the value of the start state under the behavior policy next to the true value under
    the target policy. These two numbers are the entire motivation for the assignment: averaging
    the returns as they come out of the simulator answers a different question from the one I
    was asked, and the gap between 1 and 2 shows how badly wrong that answer would be.
    """
    behaviorValue = computeBehaviorPolicyValue()
    printSectionHeading("Why the correction is needed")
    print(f"Value of A under the behavior policy b : {behaviorValue:.4f}")
    print(f"Value of A under the target policy pi  : {TRUE_VALUE:.4f}")
    print("Averaging the raw returns would converge to the first number, not the second.")


def printStatistics(label, estimates):
    """
    I report the mean and standard deviation for one estimator, together with how far the mean
    sits from the true value. I include that last column because a mean on its own does not say
    whether the estimator is centred correctly, and the bias question is one of the five I have
    to answer in writing.
    """
    meanValue, standardDeviation = computeStatistics(estimates)
    print(f"{label}")
    print(f"  Mean                  = {meanValue:.6f}")
    print(f"  Standard deviation    = {standardDeviation:.6f}")
    print(f"  Deviation from truth  = {meanValue - TRUE_VALUE:+.6f}")
    return meanValue, standardDeviation


def printTheoryCheck(numberOfEpisodes, ordinaryStandardDeviation):
    """
    I compare the spread I measured for the ordinary estimator against the spread theory
    predicts. The variance of a single weighted return is 12 in this MDP, so across N episodes
    the estimate should have a standard deviation of the square root of 12 over N. Matching
    that number is how I convince myself the implementation is correct rather than merely
    producing a plausible looking answer.
    """
    singleEpisodeVariance = computeTheoreticalOrdinaryVariance()
    predictedStandardDeviation = (singleEpisodeVariance / numberOfEpisodes) ** 0.5

    printSectionHeading("Check against theory")
    print(f"Var(rho * G) for one episode          = {singleEpisodeVariance:.4f}")
    print(f"Predicted std of ordinary IS estimate = {predictedStandardDeviation:.6f}")
    print(f"Measured std of ordinary IS estimate  = {ordinaryStandardDeviation:.6f}")
    print("Weighted IS is exactly 2 on every run because every episode that survives the")
    print("ratio has rho = 4 and G = 2, so the ratio of the two sums cannot vary.")


def figurePath(arguments, fileName):
    """
    I join a figure name onto the output directory. It is a one line helper, but every figure
    call needs it and writing it out each time made the functions below harder to read.
    """
    return os.path.join(arguments.figuresDirectory, fileName)



def printScalingTable(scalingResults):
    """
    I print the variance scaling sweep as a table so that the numbers behind the figure appear in
    my saved console output. Every figure in my report should have its numbers traceable to the
    program's own output rather than read off a picture.
    """
    print(f"  {'Episodes':>9}  {'Measured std':>13}  {'sqrt(12/N)':>12}  {'Ratio':>7}")
    for row in scalingResults:
        ratio = row["ordinaryStd"] / row["predictedStd"]
        print(f"  {row['episodes']:>9}  {row['ordinaryStd']:>13.6f}  "
              f"{row['predictedStd']:>12.6f}  {ratio:>7.3f}")


def printBiasTable(biasResults):
    """
    I print the small sample bias study, which is the evidence behind my answer on unbiasedness.
    The weighted column climbing from about 0.5 towards 2 while the ordinary column stays flat is
    the whole result, and it is easier to quote from a table than to read off the figure.
    """
    print(f"  {'Episodes':>9}  {'Ordinary mean':>14}  {'Weighted mean':>14}")
    for row in biasResults:
        print(f"  {row['episodes']:>9}  {row['ordinaryMean']:>14.4f}  {row['weightedMean']:>14.4f}")


def printChainTable(chainResults):
    """
    I print the chain length study next to its analytical prediction. The measured variance should
    track four times two to the chain length minus four, and at length two it should reproduce the
    variance of 12 that I derived for the assignment's own MDP.
    """
    print(f"  {'Length':>7}  {'rho':>9}  {'Survival':>10}  {'Var theory':>12}  {'Var measured':>13}")
    for row in chainResults:
        print(f"  {row['chainLength']:>7}  {row['survivingRatio']:>9.0f}  "
              f"{row['measuredSurvivalRate']:>10.5f}  {row['variance']:>12.1f}  "
              f"{row['measuredVariance']:>13.1f}")


def printSoftTargetTable(softTargetResults):
    """
    I print the softened target policy study, which is the only place the weighted estimator's
    general bias is visible. I include the standard error so that the bias can be judged against
    the precision of my measurement rather than taken on trust.
    """
    print(f"  {'pi(Left)':>9}  {'True value':>11}  {'Ord bias':>10}  {'Wtd bias':>10}  {'Wtd se':>8}")
    for row in softTargetResults:
        print(f"  {row['leftProbability']:>9.2f}  {row['trueValue']:>11.3f}  "
              f"{row['ordinaryBias']:>+10.4f}  {row['weightedBias']:>+10.4f}  "
              f"{row['weightedStandardError']:>8.4f}")


def writeRequiredFigures(arguments, ordinaryEstimates, weightedEstimates):
    """
    I write the figures that Part 7 actually asks for, plus the box plot and the single run
    convergence trace that go with them. These use nothing but the assignment's own MDP and its
    own episode count, so they are the figures I would submit if nothing else existed.
    """
    print("Drawing the required figures ...")
    _, _, convergenceHistory = runSingleExperiment(arguments.episodes, trackHistory=True)

    return [
        plotEstimatesPerExperiment(ordinaryEstimates, weightedEstimates,
                                   figurePath(arguments, "estimates_per_experiment.png")),
        plotConvergence(convergenceHistory, figurePath(arguments, "convergence.png")),
        plotEstimateSpread(ordinaryEstimates, weightedEstimates,
                           figurePath(arguments, "estimate_spread.png")),
    ]


def writeSupportingFigures(arguments):
    """
    I write the extra figures that back up my written answers. Every one of them still uses the
    assignment's MDP and its two policies, so nothing here changes the problem I was set, it only
    looks at it from an angle the required plots do not cover. I print a line before each study
    because the variance scaling sweep runs several million episodes and I would rather the run
    show progress than appear to have hung.
    """
    writtenFigures = []

    print("Comparing the uncorrected average against both estimators ...")
    writtenFigures.append(plotNaiveVersusCorrected(
        runNaiveComparison(20000), figurePath(arguments, "naive_vs_corrected.png")))

    print("Checking trajectory frequencies against the policies ...")
    writtenFigures.append(plotTrajectoryDistributions(
        measureTrajectoryFrequencies(50000),
        computeTargetTrajectoryProbabilities(),
        figurePath(arguments, "trajectory_distributions.png")))

    print("Measuring the distribution of the importance ratio ...")
    importanceRatios, effectiveSampleSize = collectImportanceRatios(arguments.episodes)
    print(f"  effective sample size = {effectiveSampleSize:.0f} of {len(importanceRatios)} episodes "
          f"({effectiveSampleSize / len(importanceRatios):.1%})")
    writtenFigures.append(plotRatioDistribution(
        importanceRatios, effectiveSampleSize, figurePath(arguments, "ratio_distribution.png")))

    print("Sweeping the episode count to check the spread against theory ...")
    scalingResults = runVarianceScalingStudy(SCALING_EPISODE_COUNTS, SCALING_REPETITIONS)
    printScalingTable(scalingResults)
    writtenFigures.append(plotVarianceScaling(
        scalingResults, figurePath(arguments, "variance_scaling.png")))

    print("Building the sampling distribution of the ordinary estimator ...")
    writtenFigures.append(plotSamplingDistribution(
        runSamplingDistributionStudy(1000, 1000), 1000,
        figurePath(arguments, "sampling_distribution.png")))

    print("Measuring bias at small sample sizes ...")
    biasResults = runBiasStudy(BIAS_EPISODE_COUNTS, 4000)
    printBiasTable(biasResults)
    writtenFigures.append(plotBiasVersusEpisodes(
        biasResults, figurePath(arguments, "bias_vs_episodes.png")))

    print("Tracking convergence across many independent runs ...")
    checkpoints, ordinarySummary, weightedSummary = runConvergenceEnvelope(200, arguments.episodes)
    writtenFigures.append(plotConvergenceEnvelope(
        checkpoints, ordinarySummary, weightedSummary,
        figurePath(arguments, "convergence_envelope.png")))

    return writtenFigures


def writeExtensionFigures(arguments):
    """
    I write the two figures that need a modified environment. The assignment's MDP caps the
    importance ratio at 4 and uses a deterministic target policy, so it cannot demonstrate either
    the exponential growth of the ratio or the bias of the weighted estimator. These figures vary
    one thing at a time and are marked on their face as extensions, so that nothing here can be
    mistaken for a required result.
    """
    print("Running the extensions: ratio growth and weighted estimator bias ...")

    chainResults = runChainLengthStudy(CHAIN_LENGTHS, 100000)
    behaviorResults = runBehaviorProbabilitySweep(BEHAVIOR_LEFT_PROBABILITIES, 50000)
    softTargetResults = runSoftTargetBiasStudy(
        SOFT_TARGET_LEFT_PROBABILITIES, SOFT_TARGET_EPISODE_COUNT, 20000)
    printChainTable(chainResults)
    printSoftTargetTable(softTargetResults)

    return [
        plotRatioExplosion(chainResults, behaviorResults,
                           figurePath(arguments, "ratio_explosion.png")),
        plotSoftTargetBias(softTargetResults, SOFT_TARGET_EPISODE_COUNT,
                           figurePath(arguments, "soft_target_bias.png")),
    ]


def main():
    """
    I run the assignment in the order the parts are listed: generate episodes and compute both
    estimators across all repetitions, summarise them, check the ordinary estimator against
    theory, and finally draw the figures. I seed once at the start rather than before each
    repetition, because the repetitions are supposed to be independent draws of the whole
    procedure and reseeding each one would make them identical.
    """
    arguments = parseArguments()
    setRandomSeed(arguments.seed)

    printSectionHeading(
        f"Running {arguments.experiments} experiments of {arguments.episodes} episodes"
    )
    ordinaryEstimates, weightedEstimates = runAllExperiments(
        numberOfExperiments=arguments.experiments,
        numberOfEpisodes=arguments.episodes,
    )

    printSectionHeading("Results over all experiments")
    _, ordinaryStandardDeviation = printStatistics("Ordinary importance sampling", ordinaryEstimates)
    print()
    printStatistics("Weighted importance sampling", weightedEstimates)

    printTheoryCheck(arguments.episodes, ordinaryStandardDeviation)
    printWhyImportanceSamplingIsNeeded()

    if arguments.noPlots:
        return

    os.makedirs(arguments.figuresDirectory, exist_ok=True)
    writtenFigures = writeRequiredFigures(arguments, ordinaryEstimates, weightedEstimates)

    if not arguments.quick:
        writtenFigures += writeSupportingFigures(arguments)
        writtenFigures += writeExtensionFigures(arguments)

    printSectionHeading("Figures written")
    for figurePath in writtenFigures:
        print(f"  {figurePath}")


if __name__ == "__main__":
    main()
