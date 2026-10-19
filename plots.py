"""
plots.py

This file produces the figures for Part 7. The required plot is each estimate against the
experiment number, which is where the difference in spread between the two estimators is
visible at a glance. I added a second figure showing the running estimate against the number
of episodes within a single experiment, because the required plot shows how far apart the two
estimators end up but not how quickly they get there, and that is what the last written
question asks about.
"""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiment import TRUE_VALUE

ORDINARY_COLOR = "#C44E52"
WEIGHTED_COLOR = "#4C72B0"
TRUTH_COLOR = "#555555"


def plotEstimatesPerExperiment(ordinaryEstimates, weightedEstimates, outputPath):
    """
    I plot both estimators against the experiment number, with a dashed line at the true value
    of 2 so that the reader can judge both accuracy and spread from the same picture. I use
    markers joined by lines because the experiment number is a discrete index rather than a
    continuous axis, and the line is only there to help the eye follow the sequence. I force
    the y limits to be symmetric around the true value so that neither estimator looks closer
    to the truth than it really is because of an auto-scaled axis.
    """
    experimentNumbers = list(range(1, len(ordinaryEstimates) + 1))

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.axhline(TRUE_VALUE, color=TRUTH_COLOR, linestyle="--", linewidth=1.2,
                 label=f"True value V(A) = {TRUE_VALUE:.1f}")
    axis.plot(experimentNumbers, ordinaryEstimates, marker="o", markersize=5,
              color=ORDINARY_COLOR, linewidth=1.2, label="Ordinary IS")
    axis.plot(experimentNumbers, weightedEstimates, marker="s", markersize=5,
              color=WEIGHTED_COLOR, linewidth=1.2, label="Weighted IS")

    largestDeviation = max(
        abs(value - TRUE_VALUE) for value in ordinaryEstimates + weightedEstimates
    )
    axisMargin = max(largestDeviation * 1.4, 0.02)
    axis.set_ylim(TRUE_VALUE - axisMargin, TRUE_VALUE + axisMargin)

    axis.set_xlabel("Experiment number")
    axis.set_ylabel("Estimated V(A)")
    axis.set_title("Ordinary and weighted importance sampling estimates across 20 experiments")
    axis.set_xticks(experimentNumbers)
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    return outputPath


def plotConvergence(history, outputPath):
    """
    I plot the running estimate against the number of episodes seen so far, for one experiment.
    I put the episode count on a log axis because almost all of the interesting movement
    happens in the first few hundred episodes and a linear axis would compress it into the left
    edge. This figure is my evidence that the weighted estimator settles on the right answer
    far earlier than the ordinary one, which is the practical convergence argument.
    """
    episodeCounts = [row[0] for row in history]
    runningOrdinary = [row[1] for row in history]
    runningWeighted = [row[2] for row in history]

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.axhline(TRUE_VALUE, color=TRUTH_COLOR, linestyle="--", linewidth=1.2,
                 label=f"True value V(A) = {TRUE_VALUE:.1f}")
    axis.plot(episodeCounts, runningOrdinary, color=ORDINARY_COLOR, linewidth=1.3,
              label="Ordinary IS")
    axis.plot(episodeCounts, runningWeighted, color=WEIGHTED_COLOR, linewidth=1.3,
              label="Weighted IS")

    axis.set_xscale("log")
    axis.set_xlabel("Episodes seen (log scale)")
    axis.set_ylabel("Running estimate of V(A)")
    axis.set_title("Convergence of both estimators within a single experiment")
    axis.grid(alpha=0.3, which="both")
    axis.legend()

    figure.tight_layout()
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    return outputPath


def plotEstimateSpread(ordinaryEstimates, weightedEstimates, outputPath):
    """
    I draw a box plot of the twenty estimates from each method side by side. The line plot
    already shows the spread, but a box plot states it directly and makes the comparison of
    variances, which is the whole point of the assignment, easy to quote in a single figure.
    I overlay the individual points so that the degenerate weighted case, where every run gives
    exactly the same number, is visible rather than appearing as an empty box. I set the tick
    labels myself instead of passing them to boxplot, because the keyword for that was renamed
    in recent matplotlib versions and setting the ticks directly works on every version.
    """
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.axhline(TRUE_VALUE, color=TRUTH_COLOR, linestyle="--", linewidth=1.2,
                 label=f"True value V(A) = {TRUE_VALUE:.1f}")

    axis.boxplot([ordinaryEstimates, weightedEstimates], widths=0.5)
    axis.set_xticks([1, 2])
    axis.set_xticklabels(["Ordinary IS", "Weighted IS"])

    for position, estimates, color in (
        (1, ordinaryEstimates, ORDINARY_COLOR),
        (2, weightedEstimates, WEIGHTED_COLOR),
    ):
        axis.scatter([position] * len(estimates), estimates, color=color, alpha=0.6, zorder=3)

    axis.set_ylabel("Estimated V(A)")
    axis.set_title("Spread of the two estimators over 20 experiments")
    axis.grid(alpha=0.3, axis="y")
    axis.legend()

    figure.tight_layout()
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    return outputPath


NAIVE_COLOR = "#8C8C00"
EXTENSION_NOTE = "Extension: not part of the specified environment"


def annotateAsExtension(figure):
    """
    I stamp a short note on every figure that comes from a modified environment. The assignment
    specifies one MDP and I did not want a reader flicking through my figures to mistake an
    extension for a required result, so each one says so on its face rather than only in the
    caption or the README.
    """
    figure.text(0.99, 0.01, EXTENSION_NOTE, ha="right", va="bottom",
                fontsize=8, style="italic", color="#777777")


def plotNaiveVersusCorrected(history, outputPath):
    """
    I plot the uncorrected average of the raw returns alongside both importance sampling
    estimates, with reference lines at 1 and 2. This is my answer to the first written question.
    The point of the figure is not that the naive average is noisy, because it is not, it is
    that the naive average converges tightly and confidently onto 1.0, which is the value of the
    behavior policy and not the answer I was asked for. A wrong answer that looks converged is a
    much harder failure to catch than a noisy one.
    """
    episodeCounts = [row[0] for row in history]

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.axhline(TRUE_VALUE, color=TRUTH_COLOR, linestyle="--", linewidth=1.2,
                 label="True target value = 2.0")
    axis.axhline(1.0, color=TRUTH_COLOR, linestyle=":", linewidth=1.2,
                 label="Behavior policy value = 1.0")

    axis.plot(episodeCounts, [row[1] for row in history], color=NAIVE_COLOR, linewidth=1.6,
              label="No correction (raw average)")
    axis.plot(episodeCounts, [row[2] for row in history], color=ORDINARY_COLOR, linewidth=1.3,
              label="Ordinary IS")
    axis.plot(episodeCounts, [row[3] for row in history], color=WEIGHTED_COLOR, linewidth=1.3,
              label="Weighted IS")

    axis.set_xscale("log")
    axis.set_xlabel("Episodes seen (log scale)")
    axis.set_ylabel("Running estimate of V(A)")
    axis.set_title("Without the importance correction the estimate converges to the wrong value")
    axis.set_ylim(0.0, 3.0)
    axis.grid(alpha=0.3, which="both")
    axis.legend(loc="lower right")

    figure.tight_layout()
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    return outputPath


def plotTrajectoryDistributions(measuredFrequencies, policyProbabilities, outputPath):
    """
    I draw two panels about the same three trajectories. The left panel checks my generator by
    putting the measured frequencies next to the ones the behavior policy implies, which is the
    evidence that everything downstream rests on a correct simulation. The right panel puts the
    behavior and target probabilities side by side, which is the mechanism behind the first
    question: the policy I sampled from and the policy I am evaluating barely overlap, so the
    raw data is drawn from the wrong distribution and has to be reweighted.
    """
    labels = list(policyProbabilities.keys())
    barPositions = range(len(labels))
    barWidth = 0.38

    figure, (leftAxis, rightAxis) = plt.subplots(1, 2, figsize=(13, 5))

    leftAxis.bar([position - barWidth / 2 for position in barPositions],
                 [measuredFrequencies[label] for label in labels],
                 barWidth, color=ORDINARY_COLOR, label="Measured")
    leftAxis.bar([position + barWidth / 2 for position in barPositions],
                 [policyProbabilities[label][0] for label in labels],
                 barWidth, color="#AAAAAA", label="Theoretical")
    leftAxis.set_title("Generator check: trajectory frequencies under b")
    leftAxis.set_ylabel("Probability")
    leftAxis.legend()

    rightAxis.bar([position - barWidth / 2 for position in barPositions],
                  [policyProbabilities[label][0] for label in labels],
                  barWidth, color=ORDINARY_COLOR, label="Behavior policy b")
    rightAxis.bar([position + barWidth / 2 for position in barPositions],
                  [policyProbabilities[label][1] for label in labels],
                  barWidth, color=WEIGHTED_COLOR, label="Target policy pi")
    rightAxis.set_title("Distribution shift: the data comes from the wrong policy")
    rightAxis.set_ylabel("Probability")
    rightAxis.legend()

    for axis in (leftAxis, rightAxis):
        axis.set_xticks(list(barPositions))
        axis.set_xticklabels(labels, rotation=12)
        axis.set_ylim(0, 1.1)
        axis.grid(alpha=0.3, axis="y")

    figure.tight_layout()
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    return outputPath


def plotRatioDistribution(importanceRatios, effectiveSampleSize, outputPath):
    """
    I plot how often each value of the importance ratio occurs. In this MDP the ratio only ever
    takes two values, so the figure is deliberately stark: three quarters of my episodes have a
    ratio of exactly zero and contribute nothing whatsoever to either estimator. I annotate it
    with the effective sample size, which converts that observation into the single number I
    quote in my written answer, namely that ten thousand episodes are worth about two and a half
    thousand once the weights are accounted for.
    """
    zeroCount = sum(1 for ratio in importanceRatios if ratio == 0.0)
    nonZeroCount = len(importanceRatios) - zeroCount

    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(["rho = 0\n(discarded)", "rho = 4\n(kept)"], [zeroCount, nonZeroCount],
                    color=["#AAAAAA", WEIGHTED_COLOR], width=0.55)

    for bar, count in zip(bars, [zeroCount, nonZeroCount]):
        axis.text(bar.get_x() + bar.get_width() / 2, count,
                  f"{count}\n({count / len(importanceRatios):.1%})",
                  ha="center", va="bottom")

    axis.set_ylabel("Number of episodes")
    axis.set_title(
        f"Only a quarter of episodes carry any weight\n"
        f"Effective sample size = {effectiveSampleSize:.0f} of {len(importanceRatios)} episodes"
    )
    axis.set_ylim(0, len(importanceRatios) * 0.95)
    axis.grid(alpha=0.3, axis="y")

    figure.tight_layout()
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    return outputPath


def plotVarianceScaling(scalingResults, outputPath):
    """
    I plot how the spread of each estimator shrinks with the number of episodes, on log axes so
    that the expected square root relationship appears as a straight line. The left panel is the
    validation: my measured standard deviation for the ordinary estimator should sit on the
    theoretical curve across every sample size, not just at ten thousand. The right panel shows
    root mean squared error, which is the comparison that matters when choosing an estimator
    because it charges for bias and variance together. The weighted estimator is exactly zero on
    both panels in this MDP, so I mark it with an annotation rather than trying to draw zero on a
    logarithmic axis, which would silently drop the series.
    """
    episodeCounts = [row["episodes"] for row in scalingResults]

    figure, (leftAxis, rightAxis) = plt.subplots(1, 2, figsize=(13, 5))

    leftAxis.plot(episodeCounts, [row["predictedStd"] for row in scalingResults],
                  color=TRUTH_COLOR, linestyle="--", linewidth=1.4,
                  label="Theory: sqrt(12 / N)")
    leftAxis.plot(episodeCounts, [row["ordinaryStd"] for row in scalingResults],
                  marker="o", color=ORDINARY_COLOR, linewidth=1.3, label="Ordinary IS (measured)")
    leftAxis.set_ylabel("Standard deviation of the estimate")
    leftAxis.set_title("Measured spread matches theory across four decades")

    rightAxis.plot(episodeCounts, [row["ordinaryRmse"] for row in scalingResults],
                   marker="o", color=ORDINARY_COLOR, linewidth=1.3, label="Ordinary IS")
    rightAxis.set_ylabel("Root mean squared error against V(A) = 2")
    rightAxis.set_title("Error against the true value")

    for axis in (leftAxis, rightAxis):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Episodes per experiment (log scale)")
        axis.grid(alpha=0.3, which="both")
        axis.legend(loc="lower left")
        axis.text(0.97, 0.93, "Weighted IS is exactly 0 at every N\n(cannot be drawn on a log axis)",
                  transform=axis.transAxes, fontsize=9, color=WEIGHTED_COLOR,
                  ha="right", va="top")

    figure.tight_layout()
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    return outputPath


def plotSamplingDistribution(estimates, numberOfEpisodes, outputPath):
    """
    I histogram the ordinary estimate across many independent experiments and overlay the normal
    density the central limit theorem predicts, with a standard deviation of the square root of
    12 over N. Matching the shape of a whole distribution is a much stronger check than matching
    a mean and a standard deviation, because an implementation could get both moments right while
    still producing the wrong distribution. I draw the density by hand from the formula rather
    than fitting one to the data, so the curve is a genuine prediction and not a restatement.
    """
    predictedStandardDeviation = (12.0 / numberOfEpisodes) ** 0.5

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(estimates, bins=25, density=True, color=ORDINARY_COLOR, alpha=0.65,
              edgecolor="white", label="Ordinary IS estimates")

    lowerBound = TRUE_VALUE - 4 * predictedStandardDeviation
    upperBound = TRUE_VALUE + 4 * predictedStandardDeviation
    densityPoints = [lowerBound + (upperBound - lowerBound) * index / 400 for index in range(401)]
    densityValues = [
        math.exp(-0.5 * ((point - TRUE_VALUE) / predictedStandardDeviation) ** 2)
        / (predictedStandardDeviation * math.sqrt(2 * math.pi))
        for point in densityPoints
    ]

    axis.plot(densityPoints, densityValues, color=TRUTH_COLOR, linewidth=1.8,
              label=f"Predicted normal, sd = sqrt(12/{numberOfEpisodes}) = {predictedStandardDeviation:.4f}")
    axis.axvline(TRUE_VALUE, color=TRUTH_COLOR, linestyle="--", linewidth=1.2,
                 label="True value V(A) = 2.0")

    axis.set_xlabel("Ordinary IS estimate")
    axis.set_ylabel("Density")
    axis.set_title(f"Sampling distribution of the ordinary estimator over {len(estimates)} experiments")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=9)

    figure.tight_layout()
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    return outputPath


def plotBiasVersusEpisodes(biasResults, outputPath):
    """
    I plot the average estimate against the number of episodes at small sample sizes, which is
    where the bias of the weighted estimator is visible in this MDP. With one episode only a
    quarter of runs contain a trajectory consistent with the target policy, so the weighted
    estimator is undefined in the rest and its average sits near 0.5. It climbs towards 2 as the
    sample grows, which is consistency. The ordinary estimator is flat on 2 throughout, which is
    unbiasedness. I draw error bars of two standard errors so that the gap at small N is clearly
    larger than the noise in my measurement of it.
    """
    episodeCounts = [row["episodes"] for row in biasResults]

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.axhline(TRUE_VALUE, color=TRUTH_COLOR, linestyle="--", linewidth=1.2,
                 label="True value V(A) = 2.0")

    axis.errorbar(episodeCounts, [row["ordinaryMean"] for row in biasResults],
                  yerr=[2 * row["ordinaryStandardError"] for row in biasResults],
                  marker="o", color=ORDINARY_COLOR, linewidth=1.3, capsize=3,
                  label="Ordinary IS (unbiased at every N)")
    axis.errorbar(episodeCounts, [row["weightedMean"] for row in biasResults],
                  yerr=[2 * row["weightedStandardError"] for row in biasResults],
                  marker="s", color=WEIGHTED_COLOR, linewidth=1.3, capsize=3,
                  label="Weighted IS (biased at small N)")

    axis.set_xscale("log")
    axis.set_xlabel("Episodes per experiment (log scale)")
    axis.set_ylabel("Mean estimate of V(A)")
    axis.set_title("Weighted IS is biased at small samples and converges to the truth as N grows")
    axis.grid(alpha=0.3, which="both")
    axis.legend(loc="lower right")

    figure.tight_layout()
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    return outputPath


def plotConvergenceEnvelope(checkpoints, ordinarySummary, weightedSummary, outputPath):
    """
    I plot the median running estimate of each method with a shaded band between the fifth and
    ninety fifth percentile across many independent runs. My single run convergence figure is
    open to the objection that I chose a lucky seed, and this figure answers that objection: the
    ordinary estimator's band is a wide funnel that narrows slowly, while the weighted
    estimator's band collapses onto the true value almost immediately and stays there.
    """
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.axhline(TRUE_VALUE, color=TRUTH_COLOR, linestyle="--", linewidth=1.2,
                 label="True value V(A) = 2.0")

    for summary, color, label in (
        (ordinarySummary, ORDINARY_COLOR, "Ordinary IS"),
        (weightedSummary, WEIGHTED_COLOR, "Weighted IS"),
    ):
        axis.fill_between(checkpoints,
                          [row[0] for row in summary],
                          [row[2] for row in summary],
                          color=color, alpha=0.22)
        axis.plot(checkpoints, [row[1] for row in summary], color=color, linewidth=1.5,
                  label=f"{label} (median, 5th to 95th percentile)")

    axis.set_xscale("log")
    axis.set_xlabel("Episodes seen (log scale)")
    axis.set_ylabel("Running estimate of V(A)")
    axis.set_title("Convergence across many independent runs, not a single seed")
    axis.set_ylim(0.0, 4.0)
    axis.text(0.015, 0.06,
              "The weighted band reaches down to zero over the first few episodes because the\n"
              "estimator is undefined until one episode survives the ratio",
              transform=axis.transAxes, fontsize=8, color="#666666")
    axis.grid(alpha=0.3, which="both")
    axis.legend(loc="upper right", fontsize=9)

    figure.tight_layout()
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    return outputPath


def plotRatioExplosion(chainResults, behaviorResults, outputPath):
    """
    I draw the two ways the importance ratio blows up, side by side, because neither can be shown
    in the assignment's own MDP where the ratio is capped at 4. The left panel stretches the chain
    so episodes get longer, and the ratio grows as two to the number of steps while the fraction
    of usable episodes halves each time. The right panel keeps episodes two steps long and instead
    makes the behavior policy take the target's action more rarely, which drives the ratio up as
    the reciprocal of that probability squared. I plot my measurements on top of the analytical
    curves so the figure doubles as a check that I derived both correctly.
    """
    figure, (leftAxis, rightAxis) = plt.subplots(1, 2, figsize=(13, 5))

    chainLengths = [row["chainLength"] for row in chainResults]
    leftAxis.plot(chainLengths, [row["survivingRatio"] for row in chainResults],
                  color=TRUTH_COLOR, linestyle="--", linewidth=1.4, label="Theory: rho = 2^T")
    leftAxis.plot(chainLengths, [row["variance"] for row in chainResults],
                  marker="o", color=ORDINARY_COLOR, linewidth=1.3, label="Var(rho G), analytical")
    leftAxis.plot(chainLengths, [row["measuredVariance"] for row in chainResults],
                  marker="x", linestyle="none", color="#000000", markersize=7,
                  label="Var(rho G), measured")
    leftAxis.set_yscale("log")
    leftAxis.set_xlabel("Episode length T (steps)")
    leftAxis.set_ylabel("Ratio and variance (log scale)")
    leftAxis.set_title("Longer episodes: rho grows as 2^T")

    leftProbabilities = [row["leftProbability"] for row in behaviorResults]
    rightAxis.plot(leftProbabilities, [row["theoreticalRatio"] for row in behaviorResults],
                   color=TRUTH_COLOR, linestyle="--", linewidth=1.4,
                   label="Theory: rho = 1 / b(Left)^2")
    rightAxis.plot(leftProbabilities, [row["measuredMaximumRatio"] for row in behaviorResults],
                   marker="x", linestyle="none", color="#000000", markersize=7,
                   label="Largest rho observed")
    rightAxis.plot(leftProbabilities, [row["variance"] for row in behaviorResults],
                   marker="o", color=ORDINARY_COLOR, linewidth=1.3, label="Var(rho G), measured")
    rightAxis.set_yscale("log")
    rightAxis.set_xlabel("Probability the behavior policy takes Left")
    rightAxis.set_ylabel("Ratio and variance (log scale)")
    rightAxis.set_title("Rarer actions: rho grows as 1 / b(Left)^2")
    rightAxis.invert_xaxis()

    for axis in (leftAxis, rightAxis):
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=9, loc="upper left")

    annotateAsExtension(figure)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    return outputPath


def plotSoftTargetBias(softTargetResults, numberOfEpisodes, outputPath):
    """
    I plot the bias of both estimators as the target policy is softened away from always choosing
    Left. I run this at a deliberately small episode count, because the weighted estimator's bias
    shrinks roughly as one over the sample size, and at the ten thousand episodes the assignment
    uses it has already decayed into the noise. At ten episodes it is tens of standard errors
    wide and impossible to mistake for sampling error.

    The figure has three readings. On the right hand edge the target policy equals the behavior
    policy, every ratio is exactly one, both estimators collapse to the same plain average and
    both biases are exactly zero, which is a sanity check that falls out of the sweep for free.
    Through the middle the two policies genuinely differ, several trajectories survive with
    different weights, and the weighted estimator sits clearly below the true value while the
    ordinary one stays on zero. On the left hand edge is the assignment's own deterministic
    target policy, where the bias comes from runs in which no episode survives the ratio at all.
    """
    leftProbabilities = [row["leftProbability"] for row in softTargetResults]

    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.axhline(0.0, color=TRUTH_COLOR, linestyle="--", linewidth=1.2, label="Zero bias")

    axis.errorbar(leftProbabilities, [row["ordinaryBias"] for row in softTargetResults],
                  yerr=[2 * row["ordinaryStandardError"] for row in softTargetResults],
                  marker="o", color=ORDINARY_COLOR, linewidth=1.3, capsize=3,
                  label="Ordinary IS")
    axis.errorbar(leftProbabilities, [row["weightedBias"] for row in softTargetResults],
                  yerr=[2 * row["weightedStandardError"] for row in softTargetResults],
                  marker="s", color=WEIGHTED_COLOR, linewidth=1.3, capsize=3,
                  label="Weighted IS")

    axis.axvline(1.0, color="#999999", linestyle=":", linewidth=1.1)
    axis.axvline(0.5, color="#999999", linestyle=":", linewidth=1.1)
    axis.text(1.0, axis.get_ylim()[1], " assignment's\n target policy", fontsize=8.5,
              color="#555555", ha="left", va="top")
    axis.text(0.5, axis.get_ylim()[1], "target = behavior,\nboth biases zero ", fontsize=8.5,
              color="#555555", ha="right", va="top")

    axis.set_xlabel("Probability the target policy takes Left")
    axis.set_ylabel("Measured bias (mean estimate minus true value)")
    axis.set_title(
        "Softening the target policy makes the weighted estimator's bias visible\n"
        f"({numberOfEpisodes} episodes per experiment, error bars are two standard errors)"
    )
    axis.invert_xaxis()
    axis.grid(alpha=0.3)
    axis.legend(loc="lower right")

    annotateAsExtension(figure)
    figure.subplots_adjust(bottom=0.16)
    figure.savefig(outputPath, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return outputPath
