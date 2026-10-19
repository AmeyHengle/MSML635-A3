# Written Answers (Part 8)

**Amey Hengle** · UID 122283961 · ameyhen@umd.edu
MSML635 — Off-Policy Monte Carlo Prediction Using Importance Sampling

All numbers quoted below come from my own run of `main.py` with seed 0, reproduced in
`results.txt`. The MDP is small enough to solve exactly, so I checked every simulated result
against a value I derived by hand.

---

## Setting up: the three trajectories

Every claim I make below follows from the fact that this MDP has only three possible
trajectories. I enumerated them before writing any code:

| Trajectory | P under behavior policy *b* | Return *G* | Ratio *ρ* |
|---|---|---|---|
| A → Right | 0.50 | 1 | 0 |
| A → Left, B → Right | 0.25 | 0 | 0 |
| A → Left, B → Left | 0.25 | 2 | 4 |

The target policy always chooses Left, so it follows A → Left → B → Left and collects a return
of 0 + 2 = 2. The true value is therefore **V(A) = 2**.

---

## Q1. Why is importance sampling necessary?

Because the policy that generated my data is not the policy I was asked to evaluate. The
episodes come from the behavior policy *b*, which flips a fair coin at each state, but the
question is about the target policy π, which always goes Left. If I simply averaged the returns
as they came out of the simulator, I would be estimating the wrong quantity:

$$v_b(A) = 0.5(1) + 0.25(0) + 0.25(2) = 1.0$$

whereas the answer I actually want is

$$v_\pi(A) = 0 + 2 = 2.0$$

So the uncorrected average converges to **1.0 instead of 2.0** — an error of 100%, not a
sampling artefact that more data would fix. My program prints both numbers side by side to make
this concrete, and `figures/naive_vs_corrected.png` shows it happening: the uncorrected average
settles tightly onto 1.0 and stays there. That is the failure mode worth internalising — it does
not look like a broken estimator, it looks like a converged one. `figures/trajectory_distributions.png`
shows the mechanism underneath: the behavior policy spreads its mass over all three trajectories
while the target policy puts everything on one.

Importance sampling fixes this by reweighting each observed return by how much more or less
likely the target policy was to produce that trajectory than the behavior policy. Trajectories
π would never take get weight 0 and drop out; trajectories π favours get weight greater than 1
and are counted more heavily.

The practical reason this matters beyond the assignment: in real problems the data is often
already collected (logged user interactions, a previously deployed controller, a human
demonstrator) and rerunning the world under the new policy is expensive, risky, or impossible.
Off-policy evaluation is what lets you ask "how would this *other* policy have done?" without
deploying it.

---

## Q2. Why can the importance sampling ratio become very large?

Because ρ is a **product** over timesteps, not a sum:

$$\rho = \prod_{t=0}^{T-1} \frac{\pi(A_t|S_t)}{b(A_t|S_t)}$$

Each individual factor is bounded above by 1 / *b*(a|s), so the whole product is bounded by
(max 1/*b*)^T. That is **exponential in the episode length**. Two things make it blow up:

1. **Long episodes.** Here T ≤ 2 and *b* = 0.5 everywhere, so ρ never exceeds 2² = 4, which is
   why this environment is well behaved. In an episode of 100 steps with the same policies, ρ
   could reach 2¹⁰⁰.
2. **Rarely-taken actions under the behavior policy.** If *b* assigns probability 0.01 to an
   action that π takes with probability 1, that single step multiplies ρ by 100.

There is a second, less obvious effect that matters just as much here. Because my target policy
is deterministic, any episode containing a single Right action gets ρ = 0 exactly. In this MDP
**75% of my episodes contribute nothing at all**, so the effective sample size behind a
10,000-episode run is only about 2,500. The probability mass piles onto fewer and fewer
trajectories, each carrying a larger weight, and the distribution of ρG becomes heavy-tailed.
That combination — most samples worthless, a few samples enormous — is the real source of
variance in importance sampling. `figures/ratio_distribution.png` quantifies it for this MDP: the
effective sample size, (Σρ)²/Σρ², comes out at almost exactly a quarter of the episode count, so
10,000 episodes are worth about 2,500.

Neither growth mechanism can be demonstrated inside the given MDP, so `figures/ratio_explosion.png`
varies one thing at a time. Lengthening the chain makes ρ grow as 2^T while the survival rate
halves at each step; making the behavior policy take Left more rarely drives ρ up as 1/b(Left)².
My measured variances land on the analytical curves in both panels. The chain at length 2 gives a
variance of exactly 12, reproducing the value I derived for the assignment's own MDP, which is my
check that the extension is a faithful generalisation rather than a different problem.

---

## Q3. Which estimator has smaller variance?

**Weighted importance sampling**, by a wide margin. From my run of 20 experiments at 10,000
episodes each:

| Estimator | Mean | Standard deviation |
|---|---|---|
| Ordinary IS | 2.007160 | 0.034730 |
| Weighted IS | 2.000000 | 0.000000 |

In this particular MDP the weighted estimator has **exactly zero** variance. That is not a
rounding artefact and not a bug — it is forced by the structure of the problem. Every episode
that survives the ratio is the same trajectory, with ρ = 4 and G = 2, so for any count *k* of
surviving episodes:

$$V_{\text{weighted}} = \frac{4 \cdot 2 \cdot k}{4 \cdot k} = 2 \quad \text{exactly}$$

There is no randomness left in the expression. I assert this as a test case
(`testWeightedEstimatorIsExactlyTrueValueInThisMdp`) with exact equality rather than a
tolerance.

The ordinary estimator does vary, and I checked its spread against theory rather than just
reporting it. The variance of a single weighted return is

$$\text{Var}(\rho G) = \mathbb{E}[(\rho G)^2] - \mathbb{E}[\rho G]^2 = 0.25(8)^2 - 2^2 = 16 - 4 = 12$$

so the estimate from N episodes should have standard deviation √(12/N) = √(12/10000) =
**0.034641**. I measured **0.034730**, which agrees to within 0.3%. This is the strongest
evidence I have that the implementation is correct.

`figures/variance_scaling.png` strengthens that check by sweeping N from 10 to 100,000: the
measured spread tracks √(12/N) across four orders of magnitude, so the estimator has the right
convergence *rate* and not merely the right value at one sample size.
`figures/sampling_distribution.png` goes further and compares the whole shape of the sampling
distribution against the normal density the central limit theorem predicts, which an
implementation could fail even while getting both moments right.

---

## Q4. Which estimator is unbiased?

**Ordinary importance sampling is unbiased.** Weighted importance sampling is biased, though
consistent (its bias vanishes as N → ∞).

Ordinary IS is unbiased because E[ρG] = v_π(s) by construction, and dividing by N — the full
episode count, *including* the episodes with ρ = 0 — leaves the expectation untouched. Checking
it here: (0.25)(4)(2) = 2, the true value. Dividing by the number of *surviving* episodes
instead would be the most natural-looking mistake and would inflate the answer fourfold, so I
wrote a test specifically for that (`testOrdinaryEstimatorDividesByEveryEpisode`).

Weighted IS is biased because it is a **ratio of two random variables**, and in general
E[X/Y] ≠ E[X]/E[Y]. The cleanest way to see the bias: with a single episode, the weighted
estimator returns that episode's return G whenever ρ > 0, regardless of how extreme ρ was — the
ρ cancels completely between numerator and denominator. Its expectation is therefore E[G | ρ > 0],
which is v_b conditioned on the trajectory being consistent with π, not v_π.

**An important caveat about this specific environment.** Weighted IS returns exactly 2.0 here,
so its measured bias is zero. That is a property of *this MDP*, not of the estimator. It happens
because the only trajectory with ρ > 0 is deterministic under π and always returns exactly 2, so
E[G | ρ > 0] = 2 = v_π(A) coincidentally. In an MDP where several distinct trajectories survive
the ratio with different returns, the bias would be visible. I note this because concluding
"weighted IS is unbiased" from this experiment alone would be the wrong lesson to draw.

I have two figures showing the bias that the headline experiment hides.

`figures/bias_vs_episodes.png` stays entirely inside the assignment's MDP and simply looks at
small sample sizes. At N = 1 only a quarter of runs contain any surviving trajectory, so the
weighted estimator averages about 0.5; it climbs through 1.15 at N = 3 and 1.54 at N = 5 before
reaching 2.0 by about N = 20. That curve is bias and consistency in one picture. The ordinary
estimator sits on 2.0 at every sample size including N = 1. The bias mechanism here is the
undefined-denominator case rather than the general ratio-estimator bias.

`figures/soft_target_bias.png` shows the general bias, and needs a modified target policy to do
it. Softening π away from deterministic lets several trajectories survive with different weights,
at which point the weighted estimator becomes a genuine ratio of random variables. At 10 episodes
per experiment its bias reaches −0.10 at π(Left) = 0.9, tens of standard errors clear of zero,
while the ordinary estimator stays on zero throughout. The sweep also contains a free sanity
check at its right-hand edge: when π(Left) = 0.5 the target policy *is* the behavior policy, every
ratio equals 1, and both estimators collapse to the same plain average with exactly zero bias.

---

## Q5. Why might weighted importance sampling converge faster in practice?

Because it is **self-normalising**, which bounds it in a way the ordinary estimator is not.

Dividing by Σρ instead of by N makes the estimate a weighted average of the returns actually
observed, so it is always inside [min G, max G]. A single enormous ρ cannot blow the estimate
up, because that same ρ appears in the denominator and largely cancels itself out. The ordinary
estimator has no such protection: one freak episode with a huge ρ enters the numerator alone and
can throw the estimate far from the truth.

My convergence figure (`figures/convergence.png`) shows this directly within a single run. The
weighted estimate locks onto 2.0 as soon as the first consistent episode arrives and never moves
again. The ordinary estimate swings between roughly 1.6 and 2.7 over the first few dozen
episodes and still has not fully settled after several hundred.

Because a single run invites the objection that I picked a favourable seed,
`figures/convergence_envelope.png` repeats the tracking over 200 independent runs and plots the
5th-to-95th percentile band. The ordinary estimator's band is a wide funnel that narrows slowly;
the weighted estimator's collapses onto the true value almost immediately. The weighted band
reaches down to zero over the first few episodes, which is the undefined-denominator case from Q4
appearing again — the same mechanism that makes it biased at small N.

The general trade-off is bias against variance. Weighted IS accepts a small bias in exchange for
a large reduction in variance, and since mean squared error is bias² + variance, the variance
term usually dominates at the sample sizes available in practice. Ordinary IS is theoretically
cleaner — it is unbiased, and its expectation is right even for N = 1 — but its variance can be
unbounded, which makes it unusable for long horizons. Sutton and Barto make the same point: the
weighted estimator is strongly preferred in practice despite the bias.

---

## Note on the provided expected results

The `ExpectedResults.txt` file quotes standard deviations of 0.18 for ordinary IS and 0.03 for
weighted IS. My means match the file exactly (2.01 and 2.00), but my standard deviations are
0.034730 and 0.000000.

I believe my numbers are correct, for two reasons:

1. My ordinary IS standard deviation matches the analytical prediction √(12/10000) = 0.034641 to
   three decimal places. A value of 0.18 would correspond to roughly 370 episodes per
   experiment, not the 10,000 the assignment specifies.
2. A weighted IS standard deviation of 0.03 is not achievable at *any* episode count in this
   MDP, because the estimator is algebraically pinned to exactly 2 whenever at least one episode
   survives the ratio. It would only be non-zero if the target policy were stochastic (for
   example 0.9 / 0.1 rather than 1.0 / 0.0), which would make the surviving ρ values differ.

The qualitative conclusion the assignment is driving at — that weighted IS has substantially
lower variance than ordinary IS — holds in my results even more strongly than in the provided
figures.


---

## Figures, and a note on scope

Twelve figures are produced by `python main.py`. Three are what Part 7 asks for; the rest I added
because each written answer above is stronger with measured evidence behind it than with an
assertion. I have grouped them by how far they sit from the assignment as specified, because I
would rather state that plainly than have a reader work it out.

**Required by Part 7**

| Figure | What it shows |
|---|---|
| `estimates_per_experiment.png` | Both estimates against experiment number |
| `convergence.png` | Running estimate within one experiment |
| `estimate_spread.png` | Box plot of the two estimators' spread |

**Additional, using the assignment's MDP and policies unchanged**

| Figure | Question | What it shows |
|---|---|---|
| `naive_vs_corrected.png` | Q1 | The uncorrected average converging to 1.0 instead of 2.0 |
| `trajectory_distributions.png` | Q1 | Generator validation, and the shift between b and π |
| `ratio_distribution.png` | Q2 | Three quarters of episodes discarded; effective sample size |
| `variance_scaling.png` | Q3 | Measured spread against √(12/N) across four decades |
| `sampling_distribution.png` | Q3 | The estimator's distribution against the CLT prediction |
| `bias_vs_episodes.png` | Q4 | Weighted IS biased at small N, converging to the truth |
| `convergence_envelope.png` | Q5 | Convergence over 200 runs rather than one seed |

**Extensions, using a modified environment**

| Figure | Question | What is changed, and why |
|---|---|---|
| `ratio_explosion.png` | Q2 | Episode length and behavior probability are varied |
| `soft_target_bias.png` | Q4 | The target policy is softened away from deterministic |

My reason for including the two extensions is that the given MDP cannot answer the questions that
were asked of it. Question 2 asks why ρ can become very large, but here it is capped at 4, because
episodes are at most two steps long and no action has probability below one half. Question 4 asks
which estimator is unbiased, but the deterministic target policy makes the weighted estimator
exactly correct and hides the bias it has in general. Answering either question from this
environment alone would mean asserting textbook facts I had not observed, and in Q4's case it
would actively invite the wrong conclusion.

So each extension varies exactly one thing, is derived analytically before being simulated, lives
in a separate module (`extensions.py`) that the required results never import, and is stamped as
an extension on the face of the figure itself. Nothing in Parts 1 to 7 depends on any of it.
