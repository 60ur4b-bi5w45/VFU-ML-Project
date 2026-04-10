# Pseudocode: Jacobian-Based Feature Entanglement Analysis for VFL Unlearning

This document presents three new algorithms — **Algorithm 5** (Feature Entanglement Analysis), **Algorithm 6** (Entanglement-Aware Passive Party Unlearning), and **Algorithm 7** (Entanglement-Aware Feature Unlearning) — designed to be fully coherent with Algorithms 1–4 from Varshney et al. (PETS 2025). All algorithms share the same notation: \(G(\cdot)\) is the embedding function, \(\theta_i\) are passive party model parameters, \(\theta_a\) is the active model, \(E_i^t\) are stored historical embeddings, and \(\mathcal{P}_1, \ldots, \mathcal{P}_k\) are passive parties.[^1][^2]

***

## Notation Reference (Consistent with Algorithms 1–4)

| Symbol | Meaning |
|--------|---------|
| \(\mathcal{P}_i\) | Passive party \(i\) |
| \(\mathcal{P}_a\) | Active party (server with labels) |
| \(\theta_i\) | Model parameters for passive party \(i\) |
| \(\theta_a\) | Active model parameters |
| \(G(\theta_i, D)\) | Embedding function: forward pass of party \(i\)'s model on minibatch \(D\) |
| \(E_i^t\) | Stored historical embeddings from party \(i\) at epoch \(t\) (stored by Algorithm 1, Line 8)[^1] |
| \(h_j^{(l)}\) | Hidden activation at layer \(l\) of party \(j\)'s model |
| \(X_k\) | Input features belonging to party \(k\) |
| \(J_{k \to j}^{(l)}\) | Cross-party Jacobian: \(\partial h_j^{(l)} / \partial X_k\) |
| \(\rho_{k,j}^{(l)}\) | Influence metric at layer \(l\) for the \((k, j)\) pair |
| \(r\) | Rank for low-rank SVD approximation |
| \(\tau_{\text{low}}, \tau_{\text{high}}\) | Classification thresholds (default 0.15, 0.40) |

***

## Algorithm 5: Feature Entanglement Analysis via Jacobian-Based Influence Matrix

This algorithm runs as a **pre-unlearning diagnostic** — before executing Algorithm 2 (Passive Party Unlearning) or Algorithm 3 (Feature Unlearning), it quantifies how much the target party's features are entangled with each remaining party's encoder.[^1]

```
Algorithm 5: Feature Entanglement Analysis in VFL
─────────────────────────────────────────────────────────
Input:
    Target party P_k (party requesting unlearning),
    Remaining passive parties {P_j : j ≠ k},
    Active party P_a,
    Trained models θ_1, ..., θ_k, θ_a (from Algorithm 1),
    Stored historical embeddings {E_i^t} (from Algorithm 1, Line 8),
    SVD rank r,
    Thresholds τ_low = 0.15, τ_high = 0.40,
    Number of sample batches N_b

Output:
    Influence matrix I ∈ ℝ^{(k-1)},
    Risk classification for each remaining party P_j

 1:  Initialize influence matrix I[j] ← 0 for all j ≠ k
 2:  for each remaining passive party P_j where j ≠ k do
 3:  │   Let L_j = number of layers in P_j's model θ_j
 4:  │   Initialize ρ_list ← []            ▷ store per-layer influence scores
 5:  │   for each layer l = 1, 2, ..., L_j do
 6:  │   │   Initialize Ĵ ← 0             ▷ accumulator for batch-averaged Jacobian
 7:  │   │   for b = 1, 2, ..., N_b do
 8:  │   │   │   Sample minibatch D (consistent with Algorithm 1, Line 3)
 9:  │   │   │   ──────────────────────────────────────────────────
10:  │   │   │   Step A: Forward pass through P_j's model
11:  │   │   │   ──────────────────────────────────────────────────
12:  │   │   │   Compute h_j^{(l)} = ForwardToLayer(θ_j, D_j, l)
13:  │   │   │       where D_j = features of D belonging to party j
14:  │   │   │
15:  │   │   │   ──────────────────────────────────────────────────
16:  │   │   │   Step B: Compute cross-party Jacobian
17:  │   │   │   ──────────────────────────────────────────────────
18:  │   │   │   Compute J_{k→j}^{(l)} = ∂h_j^{(l)} / ∂X_k
19:  │   │   │       where X_k = features of D belonging to party k
20:  │   │   │       ▷ In VFL, this cross-party gradient flows through
21:  │   │   │       ▷ the active model's backpropagated gradients:
22:  │   │   │       ▷ ∂h_j^{(l)}/∂X_k = (∂h_j^{(l)}/∂∇_{e_j}ℓ) · (∂∇_{e_j}ℓ/∂e_k) · (∂e_k/∂X_k)
23:  │   │   │       ▷   where ∂∇_{e_j}ℓ/∂e_k comes from the active model
24:  │   │   │
25:  │   │   │   Accumulate: Ĵ ← Ĵ + J_{k→j}^{(l)}
26:  │   │   end for
27:  │   │   Ĵ ← Ĵ / N_b                  ▷ batch-averaged Jacobian
28:  │   │
29:  │   │   ──────────────────────────────────────────────────────
30:  │   │   Step C: Low-rank SVD approximation
31:  │   │   ──────────────────────────────────────────────────────
32:  │   │   Compute rank-r SVD: Ĵ ≈ U_r Σ_r V_r^T
33:  │   │       where Σ_r = diag(σ_1, σ_2, ..., σ_r), σ_1 ≥ σ_2 ≥ ... ≥ σ_r
34:  │   │
35:  │   │   ──────────────────────────────────────────────────────
36:  │   │   Step D: Compute ρ metric
37:  │   │   ──────────────────────────────────────────────────────
38:  │   │   ρ_{k,j}^{(l)} = σ_1 / √(Σ_{i=1}^{r} σ_i²)
39:  │   │       ▷ ratio of dominant singular value to Frobenius norm
40:  │   │       ▷ ρ → 1: concentrated influence (high entanglement)
41:  │   │       ▷ ρ → 0: diffuse/negligible influence (low entanglement)
42:  │   │
43:  │   │   Append ρ_{k,j}^{(l)} to ρ_list
44:  │   end for
45:  │
46:  │   ──────────────────────────────────────────────────────────
47:  │   Step E: Aggregate across layers
48:  │   ──────────────────────────────────────────────────────────
49:  │   I[j] = median(ρ_list)              ▷ Influence_{k→j}
50:  │
51:  │   ──────────────────────────────────────────────────────────
52:  │   Step F: Classify risk level
53:  │   ──────────────────────────────────────────────────────────
54:  │   if I[j] < τ_low then
55:  │   │   Risk[j] ← LOW                  ▷ Unlearning safe for P_j
56:  │   else if I[j] < τ_high then
57:  │   │   Risk[j] ← MEDIUM               ▷ Monitor P_j post-unlearning
58:  │   else
59:  │   │   Risk[j] ← HIGH                 ▷ P_j likely needs local retraining
60:  │   end if
61:  end for
62:
63:  return I, Risk
```

### Computational Complexity

- Per (k, j, l) triple: \(O(N_b \cdot d_{h} \cdot d_{X} + r \cdot (d_{h} + d_{X}))\) where \(d_h = \dim(h_j^{(l)})\), \(d_X = \dim(X_k)\)
- Total: \(O((k-1) \cdot L_{\max} \cdot N_b \cdot d_h \cdot d_X)\)
- With randomized SVD (rank \(r \ll \min(d_h, d_X)\)): reduces to \(O(r \cdot (d_h + d_X))\) per Jacobian[^3]

### Privacy-Aware Variant (Lines 18–23)

In a strict VFL setting, the active party cannot directly access \(X_k\) or \(h_j^{(l)}\). The cross-party Jacobian is computed indirectly via the **chain rule through stored embeddings**:[^4]

\[\frac{\partial h_j^{(l)}}{\partial X_k} = \underbrace{\frac{\partial h_j^{(l)}}{\partial \nabla_{e_j} \ell}}_{\text{party } j \text{ computes}} \cdot \underbrace{\frac{\partial \nabla_{e_j} \ell}{\partial e_k}}_{\text{active party computes}} \cdot \underbrace{\frac{\partial e_k}{\partial X_k}}_{\text{party } k \text{ computes}}\]

Each factor is computed locally by the respective party, and only the matrix products (not raw data) are communicated. This preserves the privacy constraints of VFL while enabling cross-party influence estimation.

***

## Algorithm 6: Entanglement-Aware Passive Party Unlearning

This extends **Algorithm 2** (VFU-KD Passive Party Unlearning) by running Algorithm 5 as a pre-check and conditionally triggering retraining for high-risk parties.[^1]

```
Algorithm 6: Entanglement-Aware Passive Party Unlearning in VFL
─────────────────────────────────────────────────────────────────
Input:
    Target passive party P_k,
    Active party P_a,
    Learning rate η_1, distillation rate λ, current epoch t_u,
    SVD rank r, thresholds τ_low, τ_high,
    Passive party retraining learning rate η_2,
    Retraining epochs E_retrain

Output:
    Unlearned active model θ_a',
    Updated historical embeddings,
    Retrained passive models (if needed)

 ╔══════════════════════════════════════════════════════════════╗
 ║  PHASE 1: Pre-Unlearning Entanglement Diagnosis             ║
 ╚══════════════════════════════════════════════════════════════╝

 1:  (I, Risk) ← Algorithm 5(P_k, {P_j : j ≠ k}, P_a, θ_1..θ_k, θ_a,
                               {E_i^t}, r, τ_low, τ_high, N_b)

 2:  Log entanglement report:
     for each j ≠ k do
         Print: "Influence P_k → P_j: {I[j]:.3f}, Risk: {Risk[j]}"
     end for

 3:  Partition remaining parties:
     S_safe   ← {j : Risk[j] = LOW}
     S_monitor ← {j : Risk[j] = MEDIUM}
     S_retrain ← {j : Risk[j] = HIGH}

 ╔══════════════════════════════════════════════════════════════╗
 ║  PHASE 2: Execute VFU-KD (Algorithm 2)                      ║
 ╚══════════════════════════════════════════════════════════════╝

 4:  θ_a' ← Algorithm 2(P_k, P_a, η_1, λ, t_u)
         ▷ Standard passive party unlearning via knowledge distillation
         ▷ Lines 1-12 of Algorithm 2 execute as-is

 ╔══════════════════════════════════════════════════════════════╗
 ║  PHASE 3: Post-Unlearning Monitoring & Mitigation           ║
 ╚══════════════════════════════════════════════════════════════╝

     ─── For MEDIUM-risk parties (monitor) ───
 5:  for each j ∈ S_monitor do
 6:  │   Evaluate party j's contribution accuracy:
 7:  │   │   e_j = G(θ_j, D_val)           ▷ embeddings on validation set
 8:  │   │   ŷ = f(θ_a', [e_1,...,e_{j},...,e_k \ e_k])
 9:  │   │   acc_j = Accuracy(ŷ, y)
10:  │   if acc_j drops > δ_monitor (e.g., 5%) then
11:  │   │   Move j to S_retrain            ▷ escalate to retraining
12:  │   end if
13:  end for

     ─── For HIGH-risk parties (retrain) ───
14:  for each j ∈ S_retrain do
15:  │   ▷ Local retraining using KD (analogous to Algorithm 3)
16:  │   ▷ Party j retrains its encoder to adapt to the new active model
17:  │   Initialize θ_j' from θ_j           ▷ warm start, not random
18:  │   for epoch = 1, ..., E_retrain do
19:  │   │   Sample minibatch D
20:  │   │   e_j' = G(θ_j', D_j)           ▷ new embeddings
21:  │   │   Send e_j' to active party
22:  │   │   Receive ∂ℓ/∂e_j' from active party
23:  │   │   θ_j' ← θ_j' - η_2 · ∂ℓ/∂θ_j' ▷ standard VFL gradient update
24:  │   end for
25:  │   θ_j ← θ_j'
26:  end for

27:  return θ_a', {θ_j : j ∈ S_retrain}
```

### Integration with Algorithm 1

Phase 3 (Lines 14–26) reuses the same gradient communication protocol as Algorithm 1 (Lines 12–15): the active party sends \(\partial \ell / \partial e_j\) and the passive party updates locally. The key difference is that only HIGH-risk parties incur this communication cost, whereas Algorithm 2's baseline R2S approach requires **all** remaining parties to communicate.[^2][^1]

***

## Algorithm 7: Entanglement-Aware Feature Unlearning

This extends **Algorithm 3** (VFU-KD Feature Unlearning) with entanglement analysis. Since feature unlearning happens at the passive party level without active party communication, the Jacobian analysis checks whether removing target features \(F_k\) from party \(P_k\) will disrupt the **active model's** learned joint representation.[^1]

```
Algorithm 7: Entanglement-Aware Feature Unlearning in VFL
──────────────────────────────────────────────────────────
Input:
    Target passive party P_k, target features F_k,
    Learning rate η_2, current epoch t_u,
    SVD rank r, thresholds τ_low, τ_high,
    Active model θ_a (for active-side analysis)

Output:
    Unlearned passive model θ_k',
    Impact assessment on active model

 ╔══════════════════════════════════════════════════════════════╗
 ║  PHASE 1: Feature-Level Entanglement Diagnosis              ║
 ╚══════════════════════════════════════════════════════════════╝

 1:  ── Intra-party analysis: how do target features affect P_k's own encoder ──
 2:  for each layer l = 1, ..., L_k do
 3:  │   Compute J_intra^{(l)} = ∂h_k^{(l)} / ∂F_k
 4:  │       ▷ Jacobian of P_k's activations w.r.t. target features only
 5:  │   SVD: J_intra^{(l)} ≈ U_r Σ_r V_r^T
 6:  │   ρ_intra^{(l)} = σ_1 / √(Σ σ_i²)
 7:  end for
 8:  ρ_self = median({ρ_intra^{(l)} : l = 1..L_k})

 9:  ── Cross-party analysis: how do target features affect other parties ──
10:  for each remaining party P_j where j ≠ k do
11:  │   for each layer l = 1, ..., L_j do
12:  │   │   Compute J_{F_k→j}^{(l)} = ∂h_j^{(l)} / ∂F_k
13:  │   │       ▷ Same as Algorithm 5, but restricted to features F_k ⊂ X_k
14:  │   │   SVD: J_{F_k→j}^{(l)} ≈ U_r Σ_r V_r^T
15:  │   │   ρ_{F_k,j}^{(l)} = σ_1 / √(Σ σ_i²)
16:  │   end for
17:  │   I_feat[j] = median({ρ_{F_k,j}^{(l)} : all l})
18:  end for

19:  ── Active model analysis: how do target features affect the joint model ──
20:  for each layer l = 1, ..., L_a do
21:  │   Compute J_{F_k→a}^{(l)} = ∂h_a^{(l)} / ∂e_k^{F_k}
22:  │       ▷ where e_k^{F_k} = portion of P_k's embedding influenced by F_k
23:  │       ▷ Active party can compute this using stored E_k^t
24:  │   SVD and compute ρ_{F_k,a}^{(l)}
25:  end for
26:  I_active = median({ρ_{F_k,a}^{(l)} : all l})

 ╔══════════════════════════════════════════════════════════════╗
 ║  PHASE 2: Risk Classification & Decision                    ║
 ╚══════════════════════════════════════════════════════════════╝

27:  Classify Risk_self, Risk_active, and Risk[j] for all j
         using thresholds τ_low, τ_high (same as Algorithm 5, Lines 54-60)

28:  Generate impact report:
     │   "Self-impact (P_k encoder): ρ = {ρ_self}, Risk = {Risk_self}"
     │   "Active model impact: ρ = {I_active}, Risk = {Risk_active}"
     │   For each j ≠ k: "Cross-party P_j impact: ρ = {I_feat[j]}, Risk = {Risk[j]}"

 ╔══════════════════════════════════════════════════════════════╗
 ║  PHASE 3: Execute Feature Unlearning (Algorithm 3)          ║
 ╚══════════════════════════════════════════════════════════════╝

29:  θ_k' ← Algorithm 3(P_k, F_k, η_2, t_u)
         ▷ Standard feature unlearning via knowledge distillation
         ▷ Lines 1-8 of Algorithm 3 execute as-is

 ╔══════════════════════════════════════════════════════════════╗
 ║  PHASE 4: Active Model Adaptation (if needed)               ║
 ╚══════════════════════════════════════════════════════════════╝

30:  if Risk_active = HIGH then
31:  │   ▷ Active model needs to adapt to P_k's new (reduced) embeddings
32:  │   ▷ Apply KD on active model using updated embeddings from P_k
33:  │   Assign θ_a as teacher
34:  │   Initialize θ_a' with input size adjusted for new e_k dimension
35:  │   for epoch = 1, ..., E_adapt do
36:  │   │   Read stored E^t for all parties
37:  │   │   Replace E_k^t with e_k' = G(θ_k', D_k \ F_k)  ▷ new embeddings
38:  │   │   Compute teacher prediction: ŷ_T = f(θ_a, [E_1^t, ..., E_k^t])
39:  │   │   Compute student prediction: ŷ_S = f(θ_a', [E_1^t, ..., e_k', ..., E_k^t\E_k^t])
40:  │   │   Loss = (1 - λ)·ℓ(ŷ_S, y) + λ·KL(ŷ_S || ŷ_T)
41:  │   │   θ_a' ← θ_a' - η_1 · ∇_{θ_a'} Loss
42:  │   end for
43:  │   θ_a ← θ_a'
44:  end if

45:  return θ_k', θ_a, Impact Report
```

***

## How Algorithms 5–7 Relate to Algorithms 1–4

| Aspect | Algorithms 1–4 (Varshney et al.) | Algorithms 5–7 (Proposed Extension) |
|--------|----------------------------------|-------------------------------------|
| **Data dependency** | Uses stored embeddings \(E_i^t\) from Algorithm 1, Line 8 | Same — reads \(E_i^t\) for Jacobian computation |
| **Communication** | No active-passive communication during unlearning[^2] | Phase 1 (diagnosis): optional decomposed Jacobian exchange; Phase 2: no communication; Phase 3 (mitigation): communication only for HIGH-risk parties |
| **Model updates** | Gradient descent with \(\eta_1, \eta_2\) or \(H^{-1}\) | Same optimizer and learning rates; adds SVD as a diagnostic step |
| **KD mechanism** | Teacher-student with loss \((1-\lambda)\ell + \lambda \cdot \text{KL}\)[^1] | Reuses exact same KD loss for Phase 3/4 adaptation |
| **GA mechanism** | \(\theta_a \leftarrow \theta_a - \eta_1 \nabla \ell_r + \alpha \nabla \ell_f\)[^1] | Not modified — Algorithm 4 (sample unlearning) proceeds unchanged; entanglement analysis applies to party/feature unlearning only |
| **MIA auditing** | Binary classifier on output logits[^1] | Compatible — MIA audit runs after Algorithms 6/7 complete, same as after Algorithms 2/3 |

***

## Recommended Hyperparameters

These defaults align with the experimental setup in Varshney et al.:[^2]

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| SVD rank \(r\) | 10 (tabular), 20 (image) | Neural network Jacobians are naturally low-rank; top 10-20 singular values capture >95% of variance[^3] |
| \(\tau_{\text{low}}\) | 0.15 | Below this, influence is spread across many directions — minimal structured dependency |
| \(\tau_{\text{high}}\) | 0.40 | Above this, a single direction captures >40% variance — strong entanglement |
| \(N_b\) (sample batches) | 5 | Trade-off between estimation quality and computation; batch size 512 per Varshney et al.[^2] |
| \(\delta_{\text{monitor}}\) | 5% accuracy drop | Consistent with 1-5% utility loss range reported for VFU-KD[^2] |
| \(E_{\text{retrain}}\) | 10 epochs | Sufficient for encoder adaptation without full retraining |
| \(E_{\text{adapt}}\) | 5 epochs | Active model adaptation is faster since only input dimension changes |

***

## Implementation Workflow

The complete pipeline for a passive party unlearning request:

```
1. Receive unlearning request from P_k at epoch t_u

2. Run Algorithm 5 (Feature Entanglement Analysis)
   └── Outputs: Influence matrix I, Risk classification per party

3. Decision gate:
   ├── All parties LOW risk → Proceed directly to Algorithm 2
   ├── Some parties MEDIUM/HIGH → Proceed to Algorithm 6
   │   ├── Phase 2: Execute Algorithm 2 (VFU-KD)
   │   ├── Phase 3a: Monitor MEDIUM-risk parties
   │   └── Phase 3b: Retrain HIGH-risk parties (targeted communication)
   └── Log entanglement report for compliance/audit

4. Run MIA audit (Section 3.1 of original paper)
   └── Verify unlearning effectiveness as before
```

For a feature unlearning request:

```
1. Receive feature unlearning request for F_k from P_k at epoch t_u

2. Run Algorithm 7 (Entanglement-Aware Feature Unlearning)
   ├── Phase 1: Diagnose intra-party, cross-party, and active model impact
   ├── Phase 2: Classify risk levels
   ├── Phase 3: Execute Algorithm 3 (VFU-KD Feature Unlearning)
   └── Phase 4: If active model HIGH risk → adapt via KD

3. Run MIA audit
```

---

## References

1. [PETS.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/24673805/15cbb732-f33b-40f7-aa7d-21451b42bcdf/PETS.pdf?AWSAccessKeyId=ASIA2F3EMEYERY7A6LGP&Signature=Sz%2FNFDuIkWWJjrQaRI8Hosnu%2B7I%3D&x-amz-security-token=IQoJb3JpZ2luX2VjELX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIB4EM62jUcO4N53TcK1xxzcUDbOYTZqae3oQI5c9YMIeAiBM9%2FcWFZLafHkDZdsr7KB2hn%2B4iErihxD6aMSyPJUqcyrzBAh%2BEAEaDDY5OTc1MzMwOTcwNSIMqaMSBqICwqP7p7paKtAE6UsxKpIsq1uSdMTlLV406k3Rg0iJlCxT0h9imodSvFygbHHOcfSddGG8KtWgjEvgEL79XNQ4n8dcBXjyyFw4Qu7C%2FwwB9u2rMSkXt8Wc5ez3J%2BG9lsHIqNE4qyXctA7fYSvNymN7mJA9jN07vfiL5uYxUJ9iYWYUIUG%2FLTG%2Bit9n9ID6cC5x9UtdLJ3gr7bYkI0MX6ZvDzLmSWGQidl8B6zCdotSzz7mwJDCyyQTkNueTDe3xCBHCpbaDu2ikb%2FHCR6JABHXfUO4VLLuK9n2nV%2F8PtWUOLmjxxj003DSYKS9tAZ9I%2FFlPFZwxqVukRN0DPIloLAdSWH%2B6vb6%2Bz%2FABI9OnTHL96Qkbav3Wt%2FhmPMMKGWX8HxUcTBW6O%2FqUDIrRIaNHJo43iw7E888%2FQGNB9R%2BHJ8iQX0JdKx2fIaG3HXp4k%2FxJnVjpfGyipy%2BAYd%2Bhfx24Aske37PgrKwVnSLANTXc9GjOMkUe4BxR7CLBs73H9nEI7pE2CBnsKxkGnWdAJ1QSWb2xQ3%2Fs2bEVmVaubji%2Blju1uYVoef%2FbxrQLPMC35vB5bjC5zNV76VvZoRX9tEqf1vWYF0mOIVj2dAx8qtxLSlVpMdU%2BcyHuiaE9TJqjlCU1JjX5lkXuEb8njx71YwayqXQ10AVl5X3Bk6xJJRy5qm7lM8YCWaukTwRd8sqGS1P%2BthAOeb1QFejdYyhct7TaSNGlDKFNrz6EFddY1pmLHlF%2Bwd2TdzabJhqNSvv3d7lgLcC99Ai2ZFqN7zEQR0Zx4qPl0gJKqUaQtEMujCmspTNBjqZAVJvzfGvM7gekXnOWy76tXBzurpcYjgFdHgdPkkSCqp7ICUS2vdWC3ZjCjA5k%2FraHlqlJx8lptESDy9w0Ol1oc3EbguUoHCgA3fAlCwH3SgSnt9ldgYqkPgKC2Xw7l6FYzDVH2S4KlJckqmRyUn0Du9n0rO2kMlhLfe27Fr%2FUQuvx9VHmgEKbTCyh1coqbOvkQt7e1XoJoDgCA%3D%3D&Expires=1772433707) - Unlearning Clients, Features and Samples in Vertical Federated Learning

2. [Unlearning Clients, Features and Samples in Vertical Federated Learning](http://www.arxiv.org/abs/2501.13683) - Federated Learning (FL) has emerged as a prominent distributed learning paradigm. Within the scope o...

3. [Generalization Guarantees for Neural Networks via Harnessing the Low-rank Structure of the Jacobian](https://arxiv.org/abs/1906.05392v2) - Modern neural network architectures often generalize well despite containing many more parameters th...

4. [(PDF) SensFL: Privacy-Preserving Vertical Federated ...](https://file.techscience.com/files/CMES/2024/TSP_CMES-142-1/TSP_CMES_55596/TSP_CMES_55596.pdf)

