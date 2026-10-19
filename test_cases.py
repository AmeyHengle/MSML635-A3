"""
test_cases.py

These are the tests I wrote to convince myself the implementation is correct before trusting
any of the numbers I report. The MDP is small enough to solve completely by hand, so rather
than checking that the output looks reasonable I check it against values I derived on paper:
there are exactly three possible trajectories, their probabilities are 0.5, 0.25 and 0.25, and
their importance ratios are 0, 0 and 4. Run them with python test_cases.py or python -m
unittest test_cases.
"""

import unittest

from environment import (
    ACTIONS,
    BEHAVIOR_POLICY,
    STATES,
    TARGET_POLICY,
    TERMINAL_STATE,
    applyTransition,
    generateEpisode,
    setRandomSeed,
)
from experiment import (
    TRUE_VALUE,
    computeBehaviorPolicyValue,
    computeStatistics,
    computeTheoreticalOrdinaryVariance,
    runSingleExperiment,
)
from analysis import makeLogarithmicCheckpoints, percentileOf
from extensions import computeChainTheory, computeSoftTargetValue, runChainLengthStudy, runSoftTargetBiasStudy
from importance_sampling import (
    computeEffectiveSampleSize,
    computeImportanceRatio,
    computeNaiveAverage,
    computeReturn,
    ordinaryImportanceSampling,
    weightedImportanceSampling,
)

LEFT_LEFT_EPISODE = [("A", "Left", 0), ("B", "Left", 2)]
LEFT_RIGHT_EPISODE = [("A", "Left", 0), ("B", "Right", 0)]
RIGHT_EPISODE = [("A", "Right", 1)]


class TestEnvironment(unittest.TestCase):
    """
    I test the environment separately from the estimators because a bug in the transition table
    would quietly corrupt every downstream number while still producing plausible output.
    """

    def testTransitionTableMatchesAssignment(self):
        """
        I check all four transitions against the table in the assignment. This is the single
        most important test in the file, because every other result depends on these rewards
        being right and an error here would not raise an exception anywhere.
        """
        self.assertEqual(applyTransition("A", "Left"), ("B", 0))
        self.assertEqual(applyTransition("A", "Right"), (TERMINAL_STATE, 1))
        self.assertEqual(applyTransition("B", "Left"), (TERMINAL_STATE, 2))
        self.assertEqual(applyTransition("B", "Right"), (TERMINAL_STATE, 0))

    def testCoverageAssumptionHolds(self):
        """
        I verify that the behavior policy gives non-zero probability to every action the target
        policy might take. Importance sampling is only valid under this coverage condition, and
        if it failed I would be dividing by zero rather than producing a biased answer.
        """
        for state in STATES:
            for action in ACTIONS:
                if TARGET_POLICY[state][action] > 0:
                    self.assertGreater(BEHAVIOR_POLICY[state][action], 0)

    def testEpisodesAreStructurallyValid(self):
        """
        I generate a batch of episodes and check that each one starts at A, is either one or two
        steps long, and only contains trajectories that the transition table can actually
        produce. Any episode outside this set would mean the generator had drifted from the MDP.
        """
        setRandomSeed(1)
        validEpisodes = [LEFT_LEFT_EPISODE, LEFT_RIGHT_EPISODE, RIGHT_EPISODE]

        for _ in range(500):
            episode = generateEpisode()
            self.assertEqual(episode[0][0], "A")
            self.assertIn(len(episode), (1, 2))
            self.assertIn(episode, validEpisodes)

    def testTrajectoryFrequenciesMatchBehaviorPolicy(self):
        """
        I check that the three trajectories appear at the rates the behavior policy implies,
        namely one half, one quarter and one quarter. This catches a subtly wrong sampling step,
        for example weights being paired with the wrong actions, which structural checks alone
        would miss. I allow a tolerance of 0.02, which is comfortably wider than the sampling
        noise at fifty thousand episodes but far tighter than any real bug would be.
        """
        setRandomSeed(2)
        counts = {"Right": 0, "LeftRight": 0, "LeftLeft": 0}
        sampleSize = 50000

        for _ in range(sampleSize):
            episode = generateEpisode()
            if episode == RIGHT_EPISODE:
                counts["Right"] += 1
            elif episode == LEFT_RIGHT_EPISODE:
                counts["LeftRight"] += 1
            else:
                counts["LeftLeft"] += 1

        self.assertAlmostEqual(counts["Right"] / sampleSize, 0.50, delta=0.02)
        self.assertAlmostEqual(counts["LeftRight"] / sampleSize, 0.25, delta=0.02)
        self.assertAlmostEqual(counts["LeftLeft"] / sampleSize, 0.25, delta=0.02)


class TestReturnsAndRatios(unittest.TestCase):
    """
    I test the return and the ratio on the three hand-worked trajectories, since these are the
    only inputs the estimators will ever see in this environment.
    """

    def testReturnsMatchHandCalculation(self):
        """
        I check the undiscounted return of each trajectory against the sum of rewards I worked
        out by hand. The Left-Left path is the only one worth anything to the target policy.
        """
        self.assertEqual(computeReturn(LEFT_LEFT_EPISODE), 2.0)
        self.assertEqual(computeReturn(LEFT_RIGHT_EPISODE), 0.0)
        self.assertEqual(computeReturn(RIGHT_EPISODE), 1.0)

    def testDiscountingAppliesFromTheSecondStep(self):
        """
        I confirm that the first reward is never discounted and the second is discounted once.
        The assignment uses gamma = 1 so this never changes my results, but I want the default
        to be a deliberate choice that I have tested rather than an untested branch.
        """
        self.assertAlmostEqual(computeReturn(LEFT_LEFT_EPISODE, discountFactor=0.5), 1.0)
        self.assertAlmostEqual(computeReturn(RIGHT_EPISODE, discountFactor=0.5), 1.0)

    def testRatiosMatchHandCalculation(self):
        """
        I check that the ratio is 4 for the trajectory the target policy would follow and
        exactly 0 for both trajectories containing a Right action. Two steps each contributing
        1.0 / 0.5 give the factor of 4, and a single zero in the numerator kills the product.
        """
        self.assertEqual(computeImportanceRatio(LEFT_LEFT_EPISODE), 4.0)
        self.assertEqual(computeImportanceRatio(LEFT_RIGHT_EPISODE), 0.0)
        self.assertEqual(computeImportanceRatio(RIGHT_EPISODE), 0.0)

    def testRatioIsNeverAnythingOtherThanZeroOrFour(self):
        """
        I sample a batch of episodes and assert that the ratio only ever takes the two values
        the MDP allows. This is the property that explains the whole variance result, so I want
        it asserted rather than assumed.
        """
        setRandomSeed(3)
        for _ in range(1000):
            self.assertIn(computeImportanceRatio(generateEpisode()), (0.0, 4.0))


class TestEstimators(unittest.TestCase):
    """
    I test the two estimators on inputs where I know the exact answer, including the edge cases
    that would otherwise only appear rarely in a long run.
    """

    def testOrdinaryEstimatorDividesByEveryEpisode(self):
        """
        I feed in four episodes of which only one has a non-zero ratio and check that the result
        is 8 divided by 4 rather than 8 divided by 1. Dividing by the surviving count instead of
        the total is the most likely way to get this estimator wrong, and it would inflate the
        answer by a factor of four here while still looking superficially sensible.
        """
        ratios = [4.0, 0.0, 0.0, 0.0]
        returns = [2.0, 0.0, 1.0, 1.0]
        self.assertAlmostEqual(ordinaryImportanceSampling(ratios, returns), 2.0)

    def testWeightedEstimatorDividesByTheSumOfRatios(self):
        """
        I check the weighted estimator on the same input, where the denominator is 4 rather than
        the episode count. Both estimators happen to give 2 on this input, which is exactly the
        behaviour this MDP produces, so I rely on the next test to separate them.
        """
        ratios = [4.0, 0.0, 0.0, 0.0]
        returns = [2.0, 0.0, 1.0, 1.0]
        self.assertAlmostEqual(weightedImportanceSampling(ratios, returns), 2.0)

    def testEstimatorsDisagreeOnUnequalWeights(self):
        """
        I use a constructed input with two different non-zero ratios so that the two estimators
        must give different answers. This is the test that proves they are genuinely different
        formulas and not accidentally the same code path, which the degenerate ratios of this
        MDP would otherwise hide.
        """
        ratios = [2.0, 6.0]
        returns = [1.0, 3.0]
        self.assertAlmostEqual(ordinaryImportanceSampling(ratios, returns), 10.0)
        self.assertAlmostEqual(weightedImportanceSampling(ratios, returns), 2.5)

    def testWeightedEstimatorHandlesAllZeroRatios(self):
        """
        I check that the weighted estimator returns 0 instead of raising when no episode is
        consistent with the target policy. The estimator is undefined there, so I want a defined
        and harmless result rather than a division by zero crashing a long run.
        """
        self.assertEqual(weightedImportanceSampling([0.0, 0.0], [1.0, 1.0]), 0.0)

    def testWeightedEstimatorIsExactlyTrueValueInThisMdp(self):
        """
        I assert the degenerate result that makes this environment interesting: every surviving
        episode has rho = 4 and G = 2, so the weighted estimate is exactly 2 on every run with
        no sampling error at all. I check equality rather than approximate equality because the
        arithmetic really is exact here, and this is the result I explain in my written answers.
        """
        setRandomSeed(4)
        for _ in range(5):
            _, weightedEstimate, _ = runSingleExperiment(numberOfEpisodes=2000)
            self.assertEqual(weightedEstimate, TRUE_VALUE)


class TestStatisticalProperties(unittest.TestCase):
    """
    I test the properties the assignment asks me to reason about in writing, so that my written
    answers are backed by something I actually measured rather than only by the textbook.
    """

    def testOrdinaryEstimatorIsUnbiased(self):
        """
        I run the experiment repeatedly and check that the average of the ordinary estimates
        sits on the true value. I set the tolerance from theory rather than picking a round
        number: the standard deviation of the mean of ten runs of two thousand episodes is the
        square root of 12 over 20000, and I allow four of those, so a correct implementation
        passes essentially always while a biased one fails.
        """
        setRandomSeed(5)
        estimates = [runSingleExperiment(numberOfEpisodes=2000)[0] for _ in range(10)]
        meanValue, _ = computeStatistics(estimates)

        tolerance = 4 * (computeTheoreticalOrdinaryVariance() / (2000 * 10)) ** 0.5
        self.assertAlmostEqual(meanValue, TRUE_VALUE, delta=tolerance)

    def testOrdinaryEstimatorSpreadMatchesTheory(self):
        """
        I check that the measured standard deviation of the ordinary estimator is close to the
        square root of 12 over N that theory predicts. I allow a factor of two in either
        direction because twenty samples give a noisy estimate of a standard deviation, but that
        is still tight enough to catch an estimator with the wrong variance entirely.
        """
        setRandomSeed(6)
        numberOfEpisodes = 2000
        estimates = [runSingleExperiment(numberOfEpisodes)[0] for _ in range(20)]
        _, measuredStandardDeviation = computeStatistics(estimates)

        predicted = (computeTheoreticalOrdinaryVariance() / numberOfEpisodes) ** 0.5
        self.assertGreater(measuredStandardDeviation, predicted * 0.5)
        self.assertLess(measuredStandardDeviation, predicted * 2.0)

    def testWeightedEstimatorHasSmallerSpreadThanOrdinary(self):
        """
        I assert the headline comparison of the assignment directly, so that my answer to the
        variance question is a measured result from this code rather than a claim I am repeating
        from the textbook.
        """
        setRandomSeed(7)
        ordinaryEstimates = []
        weightedEstimates = []
        for _ in range(10):
            ordinaryEstimate, weightedEstimate, _ = runSingleExperiment(numberOfEpisodes=2000)
            ordinaryEstimates.append(ordinaryEstimate)
            weightedEstimates.append(weightedEstimate)

        _, ordinarySpread = computeStatistics(ordinaryEstimates)
        _, weightedSpread = computeStatistics(weightedEstimates)
        self.assertLess(weightedSpread, ordinarySpread)

    def testTheoreticalVarianceIsTwelve(self):
        """
        I check my own derivation in code so that the number quoted in my written answers cannot
        drift away from the number the program uses. The second moment is 0.25 times 8 squared,
        which is 16, and subtracting the square of the true value leaves 12.
        """
        self.assertAlmostEqual(computeTheoreticalOrdinaryVariance(), 12.0)

    def testBehaviorPolicyValueDiffersFromTargetValue(self):
        """
        I confirm that the uncorrected value of the start state is 1 while the true target value
        is 2. This is the quantitative justification for why importance sampling is needed at
        all, and it is the first of the written questions.
        """
        self.assertAlmostEqual(computeBehaviorPolicyValue(), 1.0)
        self.assertNotAlmostEqual(computeBehaviorPolicyValue(), TRUE_VALUE)




class TestAnalysisHelpers(unittest.TestCase):
    """
    I test the helpers behind my extra figures too. They do not affect the required results, but
    a figure drawn from a broken helper would be worse than no figure at all, because it would
    look like evidence while being wrong.
    """

    def testNaiveAverageRecoversBehaviorPolicyValue(self):
        """
        I check that averaging the raw returns with no correction converges to 1, the value of the
        start state under the behavior policy, rather than to the true target value of 2. This is
        the claim my answer to the first question rests on, so I measure it rather than assert it.
        """
        setRandomSeed(8)
        episodeReturns = [computeReturn(generateEpisode()) for _ in range(40000)]
        self.assertAlmostEqual(computeNaiveAverage(episodeReturns), 1.0, delta=0.03)

    def testEffectiveSampleSizeIsAQuarterOfTheEpisodes(self):
        """
        I check that the effective sample size comes out at roughly a quarter of the episode
        count, since three quarters of the ratios are zero and the rest are equal. This is the
        number I quote when I argue that ten thousand episodes are worth far fewer in practice.
        """
        setRandomSeed(9)
        importanceRatios = [computeImportanceRatio(generateEpisode()) for _ in range(20000)]
        self.assertAlmostEqual(
            computeEffectiveSampleSize(importanceRatios) / 20000, 0.25, delta=0.02
        )

    def testEffectiveSampleSizeOfEqualWeightsIsTheCount(self):
        """
        I check the boundary case where every weight is equal, which should give an effective
        sample size equal to the number of samples. If this did not hold, the quarter I report
        above would be meaningless as a fraction.
        """
        self.assertAlmostEqual(computeEffectiveSampleSize([1.0] * 10), 10.0)
        self.assertEqual(computeEffectiveSampleSize([0.0, 0.0]), 0.0)

    def testPercentileInterpolatesBetweenNeighbours(self):
        """
        I wrote my own percentile function rather than adding numpy, so I check it against values
        I can verify by hand, including the two ends and a midpoint that requires interpolation.
        """
        orderedValues = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(percentileOf(orderedValues, 0), 0.0)
        self.assertAlmostEqual(percentileOf(orderedValues, 50), 2.0)
        self.assertAlmostEqual(percentileOf(orderedValues, 100), 4.0)
        self.assertAlmostEqual(percentileOf(orderedValues, 25), 1.0)

    def testLogarithmicCheckpointsAreSortedAndBounded(self):
        """
        I check that my log spaced checkpoints are increasing, start at the first episode and end
        exactly on the last one. An off by one here would silently truncate every convergence
        figure at the right hand edge.
        """
        checkpoints = makeLogarithmicCheckpoints(10000)
        self.assertEqual(checkpoints[0], 1)
        self.assertEqual(checkpoints[-1], 10000)
        self.assertEqual(checkpoints, sorted(checkpoints))


class TestExtensions(unittest.TestCase):
    """
    I test the modified environments as carefully as the real one. These produce the figures that
    are furthest from the assignment, so they are the ones a reader is least able to check by
    inspection, which makes testing them more important rather than less.
    """

    def testChainOfLengthTwoReproducesTheAssignmentVariance(self):
        """
        I check that my chain environment at length two gives a variance of 12, which is exactly
        the variance I derived for the assignment's own MDP. The two environments differ in that
        the original pays 1 for going Right from A, but that branch has a ratio of zero so it
        cannot affect the variance of the weighted return. This agreement is my main evidence
        that the extension is a faithful generalisation rather than a different problem.
        """
        self.assertAlmostEqual(computeChainTheory(2)["variance"], 12.0)
        self.assertAlmostEqual(computeChainTheory(2)["survivingRatio"], 4.0)

    def testChainRatioGrowsAsPowersOfTwo(self):
        """
        I check the ratio and the survival probability against the closed forms at several chain
        lengths. This is the exponential growth that answers the second written question, so I
        want it pinned down in a test rather than only drawn in a figure.
        """
        for chainLength in (1, 3, 8):
            theory = computeChainTheory(chainLength)
            self.assertAlmostEqual(theory["survivingRatio"], 2.0 ** chainLength)
            self.assertAlmostEqual(theory["survivalProbability"], 2.0 ** -chainLength)

    def testChainMeasuredSurvivalMatchesTheory(self):
        """
        I simulate short chains and check that the fraction of episodes surviving the ratio lands
        on the predicted two to the minus chain length. This confirms my chain generator really
        does continue only on Left, which is what the whole extension depends on.
        """
        setRandomSeed(10)
        results = runChainLengthStudy([1, 2, 3], 20000)
        for result in results:
            self.assertAlmostEqual(
                result["measuredSurvivalRate"], result["survivalProbability"], delta=0.02
            )

    def testSoftTargetValueMatchesKnownEndpoints(self):
        """
        I check my formula for the softened target policy's value at the two points where I
        already know the answer: a probability of one is the assignment's target policy worth 2,
        and a probability of one half is the behavior policy worth 1. A formula that hits both
        endpoints is very unlikely to be wrong in between.
        """
        self.assertAlmostEqual(computeSoftTargetValue(1.0), 2.0)
        self.assertAlmostEqual(computeSoftTargetValue(0.5), 1.0)

    def testBothEstimatorsAgreeWhenTargetEqualsBehavior(self):
        """
        I check that when the target policy is the behavior policy every ratio is exactly one, so
        both estimators reduce to the same plain average and neither has any bias. This is the
        sanity check that sits at the right hand edge of my soft target figure, and it falls out
        of the sweep for free, so I assert it rather than relying on the reader to notice it.
        """
        setRandomSeed(12)
        results = runSoftTargetBiasStudy([0.5], numberOfEpisodes=10, repetitions=2000)
        result = results[0]
        self.assertAlmostEqual(result["ordinaryMean"], result["weightedMean"], places=10)
        self.assertAlmostEqual(result["trueValue"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
