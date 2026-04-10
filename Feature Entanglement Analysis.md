# Jacobian-Based Feature Entanglement Analysis for VFL Unlearning: A Comprehensive Review

## Executive Summary

The proposed Jacobian-Based Influence Matrix for analyzing feature entanglement in Vertical Federated Learning (VFL) unlearning is a **novel contribution** with no existing paper implementing this exact method. However, every individual component — Jacobian sensitivity analysis, SVD-based low-rank approximation, and feature entanglement in unlearning — has strong theoretical and empirical grounding in the literature. This report synthesizes the closest related works, explains why the approach is scientifically sound, assesses its utility, and provides a detailed intuitive explanation of the method.

***

## Does Any Existing Paper Implement This?

**No paper currently implements the exact Jacobian-Based Influence Matrix with the proposed \(\rho_{k,j}^{(l)}\) metric for cross-party feature entanglement analysis in VFL unlearning.** This makes it a genuinely novel contribution. However, several papers implement closely related ideas:

| Paper | Year/Venue | What It Does | How It Differs |
|-------|-----------|--------------|----------------|
| **SensFL** | CMES 2025 | Uses Jacobian L2 norm \(\|\partial e_n / \partial x_n\|_2\) for embedding-to-input sensitivity | Measures *same-party* sensitivity for privacy defense, not *cross-party* influence[^1] |
| **Ferrari** | NeurIPS 2024 | Feature sensitivity via Lipschitz continuity for federated feature unlearning | Operates in HFL, not VFL; measures model output sensitivity, not inter-party embedding dependency[^2][^3] |
| **MeGU** | 2026 | Analyzes feature-pattern entanglement showing shared "associated features" between concepts | Centralized unlearning; uses MLLM similarity estimation, not Jacobian SVD[^4][^5] |
| **Retain-Forget Entanglement** | ICLR 2026 submission | Directly addresses correlated-feature damage during unlearning | Uses augmented Lagrangian + Wasserstein-2 projection, not Jacobian-based[^6] |
| **Oymak et al.** | NeurIPS 2019 | Proves Jacobian of neural networks has low-rank structure with "information" and "nuisance" spaces | Theoretical foundation for SVD approach but applied to generalization, not unlearning[^7][^8] |
| **Certified VFU** | arXiv 2025 | First certified unlearning framework for VFL; notes "cascading impacts across all parties' sub-models" | Uses confidence matrix updates, not Jacobian analysis[^9][^10] |
| **LASER-VFL** | ICLR 2025 | Handles missing features in VFL; shows that removing a party's features degrades model | Validates the concern empirically but provides no Jacobian-based diagnostic[^11][^12] |

The key gap: existing works either (a) use Jacobian for *same-party* privacy defense, (b) address feature entanglement *without* Jacobian analysis, or (c) work in centralized/HFL settings. The proposed method uniquely combines cross-party Jacobian SVD with VFL unlearning impact assessment.

***

## Why Jacobian SVD for Cross-Party Feature Influence? (Theoretical Foundation)

### The Low-Rank Jacobian Property

Neural network Jacobians naturally exhibit **low-rank structure** — a few large singular values dominate while most are near zero. Oymak et al. (2019) formally decompose the Jacobian spectrum into an "information space" (top singular values capturing meaningful data relationships) and a "nuisance space" (small singular values capturing noise). More recent work confirms this holds across architectures and deepens during training, with depth-induced exponential scaling of ordered singular values.[^13][^7][^8]

This property is critical: it means a **low-rank SVD approximation** \(J \approx U\Sigma V^T\) with small rank \(r\) captures the dominant feature dependencies while drastically reducing memory and computation.

### Jacobian as Sensitivity Measure in VFL

SensFL (CMES 2025) provides the most direct evidence that Jacobian norms quantify feature-to-embedding sensitivity in VFL. Their formulation computes:[^1]

\[J_n = \left\| \frac{\partial e_n}{\partial x_n} \right\|_2\]

for each client \(n\), where \(e_n\) is the embedding and \(x_n\) is the input. Adding this as a regularization term during training reduces the information content of embeddings, effectively defending against reconstruction attacks while maintaining primary task accuracy (less than 5% drop). This validates that the Jacobian matrix captures meaningful feature-to-representation dependencies.[^1]

### Why Cross-Party Extension Is Needed

SensFL only measures how a party's *own* features influence *its own* embeddings. In VFL unlearning, the critical question is different: **how does removing party \(k\)'s features affect party \(j\)'s encoder representations?** This cross-party dependency arises because:

- The active model learns joint representations from concatenated embeddings[^14]
- During backpropagation, gradients flow from the shared loss through all embeddings, creating implicit coupling[^1]
- If features across parties are correlated (e.g., income from a bank and spending from a retailer), removing one party's features invalidates the statistical regularities that another party's encoder has learned to exploit

LASER-VFL (ICLR 2025) empirically demonstrates this: when a client's feature block goes missing, model accuracy can drop by 18%+ even for clients whose features remain available. This confirms the feature entanglement hypothesis but provides no diagnostic tool to predict it — exactly the gap the proposed method addresses.[^12]

***

## Detailed Explanation of the Proposed Method

### Step-by-Step Intuition

**Goal:** Before unlearning party \(k\)'s features, predict whether party \(j\)'s encoder will be harmed.

**Core Idea:** Measure how strongly party \(k\)'s input features \(X_k\) influence party \(j\)'s hidden layer activations \(h_j^{(l)}\) via the cross-party Jacobian.

### Step 1: Compute the Cross-Party Jacobian

At each layer \(l\) of party \(j\)'s model, compute:

\[\frac{\partial h_j^{(l)}}{\partial X_k}\]

This matrix has dimensions \((\text{dim of } h_j^{(l)}) \times (\text{dim of } X_k)\). Each entry tells how much a small change in one of party \(k\)'s features changes one of party \(j\)'s hidden activations.

**Intuition:** Think of this as a "sensitivity map" — it shows which of party \(k\)'s features party \(j\)'s encoder has learned to be sensitive to, even though they are in different organizations.

**How this works in VFL:** In the forward pass, party \(k\)'s features go through party \(k\)'s bottom model to produce embedding \(e_k\). This embedding is concatenated with others at the active party, and the loss backpropagates gradients back to all parties. Over many training rounds, party \(j\)'s encoder implicitly adapts to the joint distribution that includes \(X_k\)'s information (via shared gradient signals). The Jacobian captures this implicit coupling.

### Step 2: SVD and the \(\rho\) Metric

Perform Singular Value Decomposition on this Jacobian:

\[\frac{\partial h_j^{(l)}}{\partial X_k} = U \Sigma V^T\]

where \(\Sigma = \text{diag}(\sigma_1, \sigma_2, \ldots, \sigma_r)\) with \(\sigma_1 \geq \sigma_2 \geq \ldots\).

The proposed metric is:

\[\rho_{k,j}^{(l)} = \frac{\| \text{top singular value} \|_2}{\| \text{all singular values} \|_F} = \frac{\sigma_1}{\sqrt{\sum_i \sigma_i^2}}\]

**What \(\rho\) measures intuitively:**

- \(\rho\) close to 1.0 → Almost all the influence is concentrated in a **single dominant direction**. Party \(k\) has a very focused, strong effect on party \(j\). Removing party \(k\) will cause a sharp, directional disruption in party \(j\)'s representations.
- \(\rho\) close to 0 → Influence is **spread diffusely** across many directions (or is negligible overall). No single feature relationship dominates. Removing party \(k\) has minimal structured impact on party \(j\).
- \(\rho\) at intermediate values → Moderate concentration — some structured dependency exists but is not overwhelming.

**Why ratio, not absolute magnitude?** The ratio normalizes by total influence magnitude. A large absolute Jacobian with evenly spread singular values (low \(\rho\)) means broad but shallow coupling — the encoder can adapt. A large Jacobian with concentrated singular values (high \(\rho\)) means the encoder has learned to rely on a specific feature direction from party \(k\), making it brittle to removal.

### Step 3: Low-Rank Approximation for Efficiency

Full Jacobian computation is expensive: \(O(\text{dim}(h_j) \times \text{dim}(X_k))\) per sample per layer. The low-rank approximation:

\[J \approx U_r \Sigma_r V_r^T\]

keeps only the top-\(r\) singular values and vectors. Since neural network Jacobians are naturally low-rank, a small \(r\) (e.g., 5-20) captures the dominant dependencies while reducing memory from \(O(d_1 \times d_2)\) to \(O(r \times (d_1 + d_2))\).[^7][^8]

### Step 4: Aggregate Across Layers

For each (party \(k\), party \(j\)) pair, compute \(\rho_{k,j}^{(l)}\) across all layers \(l\), then take:

\[\text{Influence}_{k \to j} = \text{median}(\rho_{k,j}^{(l)}) \text{ over all layers } l\]

**Why median?** It is robust to outlier layers (e.g., the first layer may have anomalously high sensitivity due to direct feature contact, while deep layers may have vanishing gradients). The median captures the "typical" dependency across the network's depth.

### Step 5: Classification Thresholds

| Influence Level | Threshold | Interpretation | Action |
|----------------|-----------|----------------|--------|
| Low | \(\rho < 0.15\) | Party \(j\)'s encoder has minimal structured dependency on \(X_k\) | Unlearning should proceed safely |
| Medium | \(0.15 \leq \rho < 0.40\) | Moderate coupling exists; some degradation possible | Monitor party \(j\)'s accuracy post-unlearning |
| High | \(\rho \geq 0.40\) | Strong, concentrated dependency — party \(j\)'s encoder relies on \(X_k\) | Warn that party \(j\) likely needs local retraining |

These thresholds would need empirical calibration across datasets and architectures, but the structure is well-motivated by spectral theory. In neural networks with low-rank Jacobians, a \(\rho \geq 0.40\) means a single direction captures over 40% of the total variance — a strong signal of concentrated dependency.

***

## Why Is This Useful?

### Practical Value for VFL Deployment

Current VFL unlearning methods (VFU-KD, VFU-GA from Varshney et al., and the certified VFU framework) operate **blindly** with respect to cross-party feature entanglement. They remove the target party's contributions but provide no warning about collateral damage to other parties. The Jacobian-based analysis fills this gap by offering:[^9][^14]

- **Pre-unlearning diagnostic:** Before executing unlearning, the system can flag which remaining parties are at risk of accuracy degradation.
- **Informed decision-making:** If party \(k\) wants to leave but the analysis shows high influence on party \(j\), the federation coordinator can plan for party \(j\)'s retraining proactively rather than discovering the problem post-hoc.
- **Threshold-based automation:** Low-influence cases can be auto-approved for unlearning, while high-influence cases trigger human review or additional mitigation steps.

### Validation from Adjacent Literature

The feature entanglement problem is increasingly recognized as a central challenge. MeGU (2026) demonstrates that unlearning one concept damages related concepts due to shared "associated features" — their experiments show 5-15% accuracy drops on retained classes when entangled features are forgotten. The ICLR 2026 submission on Retain-Forget Entanglement specifically models this as a constrained optimization problem, confirming that "retained samples closely related to the forget set can be unintentionally affected, particularly when they share correlated features".[^4][^6]

In VFL specifically, the cascading impact is acknowledged but unstudied. Wang & Wang (2025) note that "unlearning data from one party triggers cascading impacts across all parties' sub-models" but address it through confidence matrix updates rather than analyzing the entanglement itself.[^9]

***

## Challenges and Considerations

### Privacy Constraints

Computing \(\partial h_j^{(l)} / \partial X_k\) requires access to both party \(j\)'s model internals and party \(k\)'s input features, potentially violating VFL's privacy boundaries. Possible mitigations include:

- Computing the Jacobian only at the **embedding level** (using stored embeddings at the active party, consistent with VFU-KD's assumptions)[^14]
- Using **randomized SVD** or **sketching** techniques to estimate the metric without full Jacobian materialization
- Having a trusted third party or secure computation protocol perform the analysis

### Computational Cost

Even with low-rank approximation, computing cross-party Jacobians for all \(k, j\) pairs across all layers scales as \(O(k^2 \times L \times r \times n)\) where \(L\) is the number of layers, \(r\) is the approximation rank, and \(n\) is the sample count. For the typical VFL setting of 2-5 parties with simple architectures, this is manageable. For larger setups, batch sampling and randomized estimation would be essential.[^14]

### Threshold Calibration

The proposed thresholds (0.15, 0.40) are reasonable starting points but require empirical validation across different:

- Dataset types (tabular vs. image)
- Model architectures (linear vs. deep)
- Feature correlation structures (independent vs. highly correlated)
- Unlearning methods (KD-based vs. gradient ascent)

Ferrari's approach of using Lipschitz continuity as a feature sensitivity metric suggests that the thresholds may need to be dataset-dependent, as the relationship between metric values and actual accuracy impact varies significantly across domains.[^2]

***

## Positioning as a Research Contribution

This analysis positions well as a **verification/auditing extension** to existing VFL unlearning methods. The strongest framing for a paper would be:

- **Problem:** VFL unlearning methods operate without quantifying cross-party feature entanglement, risking collateral damage to remaining parties.
- **Gap:** No existing work provides a pre-unlearning diagnostic for cross-party dependency in VFL (validated by the literature survey above).
- **Method:** Jacobian-Based Influence Matrix with SVD, grounded in proven low-rank Jacobian theory and validated Jacobian sensitivity analysis in VFL.[^7][^1]
- **Evaluation:** Apply the metric before running VFU-KD/VFU-GA, then correlate predicted influence levels with actual accuracy changes in remaining parties across multiple datasets and unlearning epochs.[^14]
- **Baselines:** Compare against simpler proxies like feature correlation coefficients, mutual information estimators, and the FeaUn certified approach.[^15]

The novelty lies in the specific combination: **cross-party + Jacobian SVD + VFL unlearning context + pre-unlearning diagnostic classification.** Each component has theoretical support, but the integration is new.

---

## References

1. [(PDF) SensFL: Privacy-Preserving Vertical Federated ...](https://file.techscience.com/files/CMES/2024/TSP_CMES-142-1/TSP_CMES_55596/TSP_CMES_55596.pdf)

2. [Ferrari: Federated Feature Unlearning via Optimizing ... - OpenReview](https://openreview.net/forum?id=YxyYTcv3hp) - This paper introduces Ferrari, a framework aimed at facilitating Federated Feature Unlearning. Ferra...

3. [[2405.17462] Ferrari: Federated Feature Unlearning via Optimizing ...](https://arxiv.org/abs/2405.17462) - We then propose an effective federated feature unlearning framework called Ferrari, which minimizes ...

4. [Machine-Guided Unlearning with Target Feature Disentanglement](https://arxiv.org/html/2602.17088v1)

5. [[PDF] Machine-Guided Unlearning with Target Feature Disentanglement](https://arxiv.org/pdf/2602.17088.pdf)

6. [Machine Unlearning under Retain–Forget Entanglement](https://openreview.net/forum?id=4WMBSHHJEr) - To address this challenge, we propose a novel two-phase optimization framework specifically designed...

7. [Generalization Guarantees for Neural Networks via Harnessing the Low-rank Structure of the Jacobian](https://arxiv.org/abs/1906.05392v2) - Modern neural network architectures often generalize well despite containing many more parameters th...

8. [Generalization Guarantees for Neural Networks](https://intra.ece.ucr.edu/~oymak/low_rank.pdf)

9. [Forgetting Any Data at Any Time: A Theoretically Certified Unlearning Framework for Vertical Federated Learning](https://ar5iv.labs.arxiv.org/html/2502.17081) - Privacy concerns in machine learning are heightened by regulations such as the GDPR, which enforces ...

10. [A Theoretically Certified Unlearning Framework for Vertical ...](https://ui.adsabs.harvard.edu/abs/2025arXiv250217081W/abstract) - Privacy concerns in machine learning are heightened by regulations such as the GDPR, which enforces ...

11. [Vertical Federated Learning with Missing Features During...](https://openreview.net/forum?id=OXi1FmHGzz) - Vertical federated learning trains models from feature-partitioned datasets across multiple clients,...

12. [Vertical Federated Learning with Missing Features During Training ...](https://arxiv.org/abs/2410.22564) - We propose LASER-VFL, a vertical federated learning method for efficient training and inference of s...

13. [Why Deep Jacobian Spectra Separate: Depth-Induced Scaling and Singular-Vector Alignment](https://arxiv.org/html/2602.12384v2)

14. [Unlearning Clients, Features and Samples in Vertical Federated Learning](http://www.arxiv.org/abs/2501.13683) - Federated Learning (FL) has emerged as a prominent distributed learning paradigm. Within the scope o...

15. [Cer-FeaUn: Certified Feature Unlearning in Vertical Federated ...](https://ieeexplore.ieee.org/document/11106725/) - In this paper, we proposed Cer-FeaUn, a certified feature unlearning, trading off between the overhe...

