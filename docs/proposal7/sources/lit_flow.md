# Literature review: flow matching design choices relevant to SignNMFlow

Compiled 2026-11-19. Every entry was checked against a fetched page (arXiv abs/HTML, proceedings or conference page). When the venue came from a search-indexed proceedings listing and not from the fetched page, the entry says so. Dropped because they could not be verified from a fetched page: Huynh 2009 "Metrics for 3D Rotations" (the Springer page is behind a redirect/403).

Target paper (context): J. He, S. Tang, S. You, Y. Hao, L. Cheng, R. Hong. "Naturalness-guided Manifold Flow Matching for Sign Language Production." arXiv:2609.33339, Sep 2026. https://arxiv.org/abs/2609.33339 [verified: fetched URL]. The abstract describes "a closed-form geodesic with a learnable deviation" trained by minimising kinetic energy under a naturalness measure. Neither the abstract nor the visible page metadata mentions Metric Flow Matching.

Notation: SignNMFlow path x_t = Exp_{x0}(t Log_{x0}(x1) + t(1-t) eps_theta), with energy E[xdot^T G^{-1} xdot] and G = m(x)(S+rho I)^{-1}.

---

## A. Riemannian / Lie-group flow matching, and when the manifold helps

### A1. Lipman, Chen, Ben-Hamu, Nickel, Le. "Flow Matching for Generative Modeling." ICLR 2023.
URL: https://arxiv.org/abs/2210.02747 (ICLR 2023 status from the iclr.cc poster listing and the OpenReview PDF found in search) [verified: fetched URL]
Summary: Introduces simulation-free CNF training by regressing onto the velocity of fixed conditional probability paths. Gaussian diffusion paths are special cases, and the conditional-OT (straight-line) path trains and samples faster than diffusion paths.
Implication for SignNMFlow: This is the Euclidean baseline that SignNMFlow argues against ("linear interpolation leaves the manifold"). A fair ablation needs a straight-line CFM on the same quaternion/6D representation, with renormalisation, and the same transformer, x1-prediction and step count.

### A2. Chen, Lipman. "Flow Matching on General Geometries" (Riemannian Flow Matching, RFM). ICLR 2024.
URL: https://arxiv.org/abs/2302.03660 [verified: fetched URL]
Summary: Extends FM to Riemannian manifolds. Conditional paths are built from a premetric, which is the geodesic distance on simple manifolds and spectral distances on general ones. Training is simulation-free on simple geometries and gives SOTA on non-Euclidean datasets.
Implication for SignNMFlow: SignNMFlow's base path Exp_{x0}(t Log_{x0} x1) on (S^3)^J is exactly the RFM geodesic path. RFM already accepts any premetric, so a data-dependent premetric is a known degree of freedom. The new part of SignNMFlow is only the learned deviation and the specific metric.

### A3. De Bortoli, Mathieu, Hutchinson, Thornton, Teh, Doucet. "Riemannian Score-Based Generative Modelling." NeurIPS 2022.
URL: https://arxiv.org/abs/2202.02763 [verified: fetched URL]
Summary: Extends score-based diffusion to compact Riemannian manifolds using geodesic random walks and heat-kernel approximations. Evaluated mainly on spherical (earth-science) data.
Implication for SignNMFlow: This is the diffusion-side alternative on (S^3)^J or SO(3)^J, with a noise-to-data process that has a principled stationary source. Useful as a baseline or citation when justifying the uniform-hemisphere p0.

### A4. Leach, Schmon, Degiacomi, Willcocks. "Denoising Diffusion Probabilistic Models on SO(3) for Rotational Alignment." ICLR 2022 Workshop on Geometrical and Topological Representation Learning.
URL: https://www.iclr.cc/virtual/2022/8698 [verified: fetched URL]
Summary: Defines DDPMs on SO(3) using the isotropic Gaussian on SO(3) (IGSO(3)). It outperforms naive Euler-angle diffusion on synthetic rotation sampling and 3D alignment.
Implication for SignNMFlow: IGSO(3) and wrapped normals are the standard tangent-space Gaussians on rotations. A concentrated IGSO(3) around a mean or rest pose is a natural informative source to replace the uniform hemisphere (see D5).

### A5. Yim, Trippe, De Bortoli, Mathieu, Doucet, Barzilay, Jaakkola. "SE(3) diffusion model with application to protein backbone generation" (FrameDiff). ICML 2023.
URL: https://arxiv.org/abs/2302.02277 [verified: fetched URL]
Summary: Develops SE(3)-invariant diffusion over many rigid frames, using IGSO(3) for the rotation part. It generates designable backbones of up to 500 residues without a pretrained structure network.
Implication for SignNMFlow: This is the template for product-of-rotations generative models. It uses self-conditioning and denoised-data prediction, and its rotation loss is computed in the tangent/axis-angle space, not on raw quaternion coordinates.

### A6. Yim, Campbell, Foong, Gastegger, Jiménez-Luna, Lewis, Satorras, Veeling, Barzilay, Jaakkola, Noé. "Fast protein backbone generation with SE(3) flow matching" (FrameFlow). arXiv preprint 2023.
URL: https://arxiv.org/abs/2310.05297 (details from https://arxiv.org/html/2310.05297v2) [verified: fetched URL]
Summary: Ports FrameDiff to SE(3) flow matching, with about 5x fewer steps and about 2x designability. The network predicts clean frames, and the vector field is derived as log_{r_t}(r̂_1)/(1-t). It uses self-conditioning. At inference, rotations follow an exponential schedule kappa(t)=e^{-ct} (c=10), while training keeps the linear schedule. The 1/(1-t)^2 loss weight is clipped at t=0.9.
Implication for SignNMFlow: This validates SignNMFlow's "predict x1, re-plan the path" sampler, which is the same construction. It also supplies three cheap tricks: self-conditioning, a faster-than-linear rotation schedule at inference only, and clipped time weighting.

### A7. Bose, Akhound-Sadegh, Huguet, Fatras, Rector-Brooks, Liu, Nica, Korablyov, Bronstein, Tong. "SE(3)-Stochastic Flow Matching for Protein Backbone Generation" (FoldFlow). ICLR 2024 (Spotlight).
URL: https://arxiv.org/abs/2310.02391 [verified: fetched URL]
Summary: A family of SE(3) FM models. A base geodesic model is extended with Riemannian minibatch OT couplings (FoldFlow-OT) and a stochastic variant (FoldFlow-SFM). OT couplings give simpler, more stable flows.
Implication for SignNMFlow: Riemannian minibatch OT on (S^3)^J or SO(3)^J is a proven, orthogonal way to straighten paths. It is cheaper than learning a deviation network, and it is a necessary baseline for any "better paths" claim.

### A8. Sherry, Smets. "Flow Matching on Lie Groups." GSI 2025 (7th Int. Conf. on Geometric Science of Information).
URL: https://arxiv.org/abs/2504.00494 [verified: fetched URL]
Summary: Generalises FM to Lie groups by replacing straight lines with exponential curves. For matrix groups this reduces to matrix exp/log, and it extends to product groups.
Implication for SignNMFlow: This supports working on SO(3)^J directly with matrix exp/log instead of S^3. That removes the hemisphere/sign issue (D9) while keeping the same geodesic paths.

### A9. Geffner, Didi, Zhang, Reidenbach, Cao, Yim, Geiger, Dallago, Kucukbenli, Vahdat, Kreis. "Proteina: Scaling Flow-based Protein Structure Generative Models." ICLR 2025 (Oral).
URL: https://arxiv.org/abs/2503.00710 (details from https://arxiv.org/html/2503.00710) [verified: fetched URL]
Summary: A SOTA backbone generator that models only Cα coordinates in Euclidean space, with no SE(3)/SO(3) frames. It uses a large transformer, velocity prediction, x̂-based self-conditioning, a Beta(1.9,1) timestep distribution, CFG and autoguidance.
Implication for SignNMFlow: This is the strongest evidence that scale plus a good Euclidean representation can beat manifold-aware frame models. SignNMFlow should show that its gains come from the geometry and not from schedule or architecture, for example against a Euclidean 6D-rotation or joint-position FM with CFG and self-conditioning.

### A10. Nadar, Foti, Birdal. "PoseD-Flow: Versatile and Guided Flow Matching Model of Human Pose." CVPR 2026 (Highlight).
URL: https://cvpr.thecvf.com/virtual/2026/poster/39570 [verified: fetched URL]
Summary: PoseRFM is an RFM model of human pose on the product manifold of joint rotations. Riemannian D-Flow adds training-free guidance by differentiating through the ODE. The analysis says the dynamics depend on data covariance and manifold curvature, and the method is SOTA on pose completion, denoising and IK.
Implication for SignNMFlow: This is the closest prior work for product-of-rotations RFM for human pose and must be cited and compared. Its covariance/curvature analysis overlaps conceptually with SignNMFlow's "naturalness metric" story.

### A11. Miao, Huang, Li. "Riemannian Motion Generation: A Unified Framework for Human Motion Representation and Generation via Riemannian Flow Matching" (RMG). arXiv, Mar 2026.
URL: https://arxiv.org/abs/2603.15016 [verified: fetched URL]
Summary: Represents human motion on a product manifold (translation plus rotations) and trains with RFM using geodesic interpolation, tangent-space supervision and on-manifold ODE integration. Reports SOTA FID on HumanML3D and MotionMillion. Ablations favour the compact translation-plus-rotation factorisation, but the fetched page shows no direct Euclidean-vs-manifold head-to-head.
Implication for SignNMFlow: This is concurrent evidence that plain RFM (no learned deviation) on rotation manifolds is already strong for motion. SignNMFlow must ablate "geodesic only" against "geodesic plus learned deviation" to isolate its contribution.

---

## B. Data-dependent metrics, learned interpolants and kinetic-optimal paths (novelty overlap)

### B1. Kapuśniak, Potaptchik, Reu, Zhang, Tong, Bronstein, Bose, Di Giovanni. "Metric Flow Matching for Smooth Interpolations on the Data Manifold" (MFM). NeurIPS 2024.
URL: https://arxiv.org/abs/2405.14780 ; https://proceedings.neurips.cc/paper_files/paper/2024/hash/f381114cf5aba4e45552869863deaaa7-Abstract.html (details from https://arxiv.org/html/2405.14780v2) [verified: fetched URL]
Summary: Learns interpolants x_{t,eta} = (1-t)x0 + t x1 + t(1-t) phi_{t,eta}(x0,x1), described as "a nonlinear correction for straight interpolants". They are trained by minimising the geodesic loss E[xdot^T G(x_t; D) xdot], which the paper calls the kinetic energy w.r.t. g. The vector field is then regressed onto the learned interpolant. Metrics are diagonal: LAND, G = (diag(h(x)) + eps I)^{-1} with h_alpha(x) = sum_i (x_i^alpha - x^alpha)^2 exp(-||x-x_i||^2/2sigma^2), a kernel-weighted local variance; and RBF, G = (diag(h~(x)) + eps I)^{-1} with h~ an RBF network centred at k-means cluster centres. G is small near data and large away from it. Results: SOTA on single-cell dynamics, LiDAR navigation and unpaired image translation.
Implication for SignNMFlow: **Major novelty overlap.** SignNMFlow's "geodesic + t(1-t) learned deviation, trained by minimising kinetic energy under a data-dependent metric built from k-means prototypes and kernel-weighted local scatter" is MFM's recipe (both the LAND and the RBF/k-means variants). It is moved into the tangent space of (S^3)^J and uses a full, not diagonal, scatter matrix. SignNMFlow must cite MFM and frame its contribution as Riemannian-base MFM for rotations. **Metric direction:** MFM's cost uses G = (local variance + eps)^{-1}, which makes moving along directions of high data spread cheap. SignNMFlow's energy uses G^{-1} = (S + rho I)/m(x). If that is implemented as described, it makes motion along high-variance directions *expensive*, the opposite of MFM/LAND and of a Mahalanobis cost xdot^T (S+rho I)^{-1} xdot. Check the code and paper for this sign/inverse convention.

### B2. Shaul, Chen, Nickel, Le, Lipman. "On Kinetic Optimal Probability Paths for Generative Models." arXiv 2023 (no venue on the fetched page).
URL: https://arxiv.org/abs/2306.06626 [verified: fetched URL]
Summary: Searches Gaussian probability paths for minimum kinetic energy. Within that family the energy depends on the data only through a scalar "data separation function". Kinetic-optimal paths solve a 1D ODE, and Cond-OT becomes kinetic-optimal when n/sqrt(d) -> 0.
Implication for SignNMFlow: This is the prior "choose the path by minimising kinetic energy" argument. Note that it minimises the energy of the *marginal* path. SignNMFlow, like MFM, minimises the energy of *conditional* paths, which does not guarantee simpler marginal velocity fields. This weakens the claim that the learned deviation makes sampling easier.

### B3. Tong, Fatras, Malkin, Huguet, Zhang, Rector-Brooks, Wolf, Bengio. "Improving and generalizing flow-based generative models with minibatch optimal transport" (CFM / OT-CFM). TMLR 2024.
URL: https://arxiv.org/abs/2302.00482 [verified: fetched URL]
Summary: Generalised conditional FM with arbitrary (non-Gaussian) sources. Minibatch-OT couplings give straighter, more stable flows that approximate dynamic OT and allow faster inference.
Implication for SignNMFlow: Minibatch OT between hemisphere noise and data quaternions, per sequence or per frame, using the geodesic cost, is a near-free baseline for path quality. MFM's best variant is OT-MFM, so the coupling and the metric should be combined.

### B4. Tong, Malkin, Fatras, Atanackovic, Zhang, Huguet, Wolf, Bengio. "Simulation-free Schrödinger bridges via score and flow matching" ([SF]^2M). AISTATS 2024.
URL: https://arxiv.org/abs/2307.03672 [verified: fetched URL]
Summary: A simulation-free objective for Schrödinger bridges using entropic (minibatch) OT couplings and Brownian-bridge conditional paths, with score and flow regression.
Implication for SignNMFlow: The stochastic analogue of conditional paths. A small Brownian-bridge noise term sigma*sqrt(t(1-t))*z in the tangent space is a principled alternative to a deterministic learned deviation, and it smooths the marginal field.

### B5. Liu, Lipman, Nickel, Karrer, Theodorou, Chen. "Generalized Schrödinger Bridge Matching" (GSBM). ICLR 2024.
URL: https://arxiv.org/abs/2310.02233 (details from https://arxiv.org/html/2310.02233v2) [verified: fetched URL]
Summary: Minimises ∫E[½||u||^2 + V_t(X_t)]dt with task-specific state costs. Conditional bridges are optimised as Gaussian paths with spline-parametrised mean and std, so no SDE simulation is needed. Includes a LiDAR-manifold experiment with a data-dependent state cost (distance to a kNN tangent plane).
Implication for SignNMFlow: This is prior work on "learn the conditional path to minimise kinetic plus data-dependent cost". It uses a *state cost* instead of a metric, which is an equally principled way to encode naturalness, for example V(x) = -log density under the prototype model. Spline control points are a lightweight alternative to an MLP deviation.

### B6. Neklyudov, Brekelmans, Tong, Atanackovic, Liu, Makhzani. "A Computational Framework for Solving Wasserstein Lagrangian Flows." arXiv 2023 (no venue on the fetched page).
URL: https://arxiv.org/abs/2310.10649 [verified: fetched URL]
Summary: Unifies OT variants by changing the kinetic-energy geometry and adding potentials. The dual formulation is solved with deep nets without trajectory simulation or optimal couplings.
Implication for SignNMFlow: Gives the theory for "change the metric in the kinetic energy" at the level of marginals. It is useful framing for what SignNMFlow's metric actually optimises.

### B7. Albergo, Boffi, Vanden-Eijnden. "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions." JMLR (per the arXiv comment "JMLR Version").
URL: https://arxiv.org/abs/2303.08797 [verified: fetched URL]
Summary: Interpolants that bridge two densities with an additional latent term, with drift and score learned by quadratic regression. Optimising the interpolant recovers the Schrödinger bridge.
Implication for SignNMFlow: This establishes that learned or optimised interpolants are an established idea, so the novelty has to lie in the metric and the manifold. It also shows that adding a latent noise term to the interpolant is a principled design knob.

### B8. Zweig, Zhang, Azizi, Knowles. "Energy Guided Geometric Flow Matching" (EGGFM). arXiv, Sep 2025.
URL: https://arxiv.org/abs/2509.25230 [verified: fetched URL]
Summary: Learns a metric tensor from data via score matching and "annealed energy distillation", then uses it to keep FM trajectories near the data manifold. The authors argue that RBF/kNN-based metrics, as in MFM, suffer from the curse of dimensionality.
Implication for SignNMFlow: Direct critique of kernel/prototype metrics. With 41 joints x 4D (164-D), or per joint, the K-means scatter metric may be poorly estimated. A score- or energy-derived metric is the 2025 alternative to test.

### B9. Wang, Wu, Zhang, Liu, Sun, Wang, Zeller, Cremers. "GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis" (PDG-FM). CVPR 2026.
URL: https://arxiv.org/abs/2603.01010 [verified: fetched URL]
Summary: Data-to-data FM whose interpolation paths follow geodesics of a density-defined data manifold. A teacher-student setup distils the density geodesics into an ambient-space predictor.
Implication for SignNMFlow: More recent evidence (CVPR 2026) that density-geodesic interpolants help in a conditional generation setting. It should be cited as related work on data-dependent geodesic paths.

### B10. Albergo, Goldstein, Boffi, Ranganath, Vanden-Eijnden. "Stochastic interpolants with data-dependent couplings." ICML 2024.
URL: https://arxiv.org/abs/2310.03725 [verified: fetched URL]
Summary: Couples base and target so that base samples are drawn conditionally on the target or conditioning information. Learnable with the same square loss, and shown on super-resolution and inpainting.
Implication for SignNMFlow: Justifies replacing the uniform hemisphere source with a conditioning-dependent source, for example a pose retrieved from a gloss dictionary, a mean pose per gloss, or a noised coarse prediction. This shortens paths and makes 5-step sampling easier.

---

## C. Prediction parameterisation, weighting and schedules

### C1. Li, He. "Back to Basics: Let Denoising Generative Models Denoise" (JiT). arXiv tech report, Nov 2025.
URL: https://arxiv.org/abs/2511.13720 (details from https://arxiv.org/html/2511.13720v2) [verified: fetched URL]
Summary: Argues from the manifold assumption that predicting clean data is fundamentally easier than predicting noise or velocity in high dimensions. JiT-B/16 ablation (FID): x-prediction with v-loss 8.62, x-pred/x-loss 10.14, x-pred/eps-loss 10.45; eps- and v-prediction fail (FID > 90). Uses logit-normal t with mu=-0.8, sigma=0.8, clips (1-t) at 0.05, picks EMA decay post hoc, and uses CFG with an interval.
Implication for SignNMFlow: Supports SignNMFlow's x1-prediction choice. It also suggests computing the loss in velocity space (x-pred + v-loss, i.e. weighting by 1/(1-t)^2, clipped) instead of a plain ||x1 - x̂1||^2. That was worth about 15% FID in JiT. EMA and CFG are standard parts of the recipe.

### C2. Salimans, Ho. "Progressive Distillation for Fast Sampling of Diffusion Models." ICLR 2022.
URL: https://arxiv.org/abs/2202.00512 [verified: fetched URL]
Summary: Introduces parameterisations that stay stable at few sampling steps (including v-prediction) and progressive distillation down to about 4 steps.
Implication for SignNMFlow: At 5 steps, the parameterisation and loss weighting matter a lot. Progressive or consistency distillation of a many-step SignNMFlow teacher is a standard way to keep quality at 1 to 5 steps.

### C3. Kingma, Salimans, Poole, Ho. "Variational Diffusion Models." NeurIPS 2021.
URL: https://arxiv.org/abs/2107.00630 [verified: fetched URL]
Summary: Shows that the continuous-time VLB "is invariant to the noise schedule, except for the signal-to-noise ratio at its endpoints". The schedule can be learned to reduce variance.
Implication for SignNMFlow: Changing the path's speed profile (the t-parameterisation) is equivalent to reweighting time. Part of what a learned deviation buys may be reproducible with a schedule or weighting change, and this needs an ablation.

### C4. Kingma, Gao. "Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation." NeurIPS 2023 (Oral).
URL: https://arxiv.org/abs/2303.00848 ; https://papers.nips.cc/paper_files/paper/2023/hash/ce79fbf9baef726645bc2337abb0ade2-Abstract-Conference.html [verified: fetched URL]
Summary: Common diffusion objectives are weighted integrals of ELBOs over noise levels. Monotonic weightings are equivalent to the ELBO plus Gaussian-noise augmentation, and new weightings give SOTA FID.
Implication for SignNMFlow: x-loss, v-loss and eps-loss are just different time weightings. SignNMFlow's ||x1 - x̂1||^2 is one specific, implicit weighting that should be chosen deliberately.

### C5. Chen. "On the Importance of Noise Scheduling for Diffusion Models." arXiv tech report 2023.
URL: https://arxiv.org/abs/2301.10972 [verified: fetched URL]
Summary: The optimal schedule is task- and dimension-dependent, and higher-dimensional, more redundant data needs noisier schedules. Scaling the input (shifting log-SNR) is a simple, robust fix.
Implication for SignNMFlow: Quaternion sequences are highly redundant, both temporally and across joints. Shifting time sampling or the schedule toward the noisy end is a cheap knob that is probably under-tuned.

### C6. Esser, Kulal, Blattmann, Entezari, Müller, Saini, Levi, Lorenz, Sauer, Boesel, Podell, Dockhorn, English, Rombach. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3). ICML 2024 (Best Paper).
URL: https://arxiv.org/abs/2403.03206 ; https://icml.cc/virtual/2024/poster/34535 [verified: fetched URL]
Summary: Large-scale study of rectified-flow training choices. Logit-normal timestep sampling, lognorm(0,1), consistently ranks well and beats uniform t. The paper shows that changing the timestep distribution is equivalent to a time-dependent loss weighting, with the same optimum but different optimisation.
Implication for SignNMFlow: Replace uniform t with logit-normal or a mode-shifted density. This is a one-line change with strong evidence behind it.

---

## D. Conditional-FM training and sampling tricks, priors, few-step, rotations

### D1. Ho, Salimans. "Classifier-Free Diffusion Guidance." NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications (short version).
URL: https://arxiv.org/abs/2207.12598 [verified: fetched URL]
Summary: Trains conditional and unconditional models jointly via condition dropout and mixes their predictions at sampling to trade diversity for fidelity.
Implication for SignNMFlow: SignNMFlow does not use CFG. Text-to-sign is a strongly conditional task where CFG usually improves condition adherence (DTW/BLEU back-translation). It needs only about 10% text dropout during training.

### D2. Zheng, Le, Shaul, Lipman, Grover, Chen. "Guided Flows for Generative Modeling and Decision Making." arXiv 2023.
URL: https://arxiv.org/abs/2311.13443 [verified: fetched URL]
Summary: Brings CFG to flow-matching models and improves conditional image generation and zero-shot TTS.
Implication for SignNMFlow: Shows directly that CFG transfers to FM. With x1-prediction, guidance can be applied on x̂1, then renormalised per joint and the path re-planned. Proteina (A9) and JiT (C1) also use CFG.

### D3. Chen, Zhang, Hinton. "Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning." ICLR 2023.
URL: https://arxiv.org/abs/2208.04202 [verified: fetched URL]
Summary: Introduces self-conditioning, which feeds the model's previous x̂ estimate back as input, and finds that it substantially improves sample quality.
Implication for SignNMFlow: SignNMFlow already re-plans with x̂1 at each step, so feeding x̂1 from the previous step as an extra input is essentially free. FrameDiff, FrameFlow and Proteina all use it.

### D4. Karras, Aittala, Lehtinen, Hellsten, Aila, Laine. "Analyzing and Improving the Training Dynamics of Diffusion Models" (EDM2). arXiv 2023 (no venue on the fetched page).
URL: https://arxiv.org/abs/2312.02696 [verified: fetched URL]
Summary: Magnitude-preserving architecture redesign plus post-hoc EMA, which reconstructs any EMA length after training. EMA length interacts strongly with quality and guidance.
Implication for SignNMFlow: No EMA is mentioned. Adding EMA weights, ideally post-hoc EMA tuned on val, is a low-risk gain.

### D5. Lee, Kim, Shin, Tan, Liu, Meng, Qin, Chen, Yoon, Liu. "PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior." ICLR 2022.
URL: https://arxiv.org/abs/2106.06406 [verified: fetched URL]
Summary: Replaces the N(0,I) prior with a prior whose statistics come from the conditioning input (mel-spectrogram energy), which speeds up training and inference and improves quality, especially for small networks.
Implication for SignNMFlow: The uniform hemisphere p0 is maximally uninformative and far from natural poses. A wrapped normal or IGSO(3) centred on the per-joint Fréchet mean, or on a text-conditioned coarse pose, shortens geodesics. It also makes a data-manifold metric meaningful near t=0, where MFM-style metrics are otherwise evaluated far from any data.

### D6. Geng, Deng, Bai, Kolter, He. "Mean Flows for One-step Generative Modeling." NeurIPS 2025 (Oral).
URL: https://arxiv.org/abs/2505.13447 ; NeurIPS 2025 status per https://proceedings.nips.cc/paper_files/paper/2025/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html (search-indexed) [verified: fetched URL]
Summary: Learns the average velocity over [r,t] using an identity linking it to the instantaneous velocity. Reaches 3.43 FID at 1 NFE on ImageNet-256 from scratch.
Implication for SignNMFlow: If the 5-step budget is fixed, a Riemannian MeanFlow, with average velocity in the tangent space and Exp-map updates, is the strongest 2025 single-network few-step method to adapt.

### D7. Frans, Hafner, Levine, Abbeel. "One Step Diffusion via Shortcut Models." ICLR 2025 (Oral; venue per iclr.cc/proceedings found in search).
URL: https://arxiv.org/abs/2410.12557 [verified: fetched URL]
Summary: Conditions the network on both noise level and step size and trains self-consistency across step sizes. One network serves any step budget and beats consistency models and reflow.
Implication for SignNMFlow: An alternative to MeanFlow. Step-size conditioning adds very little to the transformer and directly targets SignNMFlow's few-step regime.

### D8. Zhou, Barnes, Lu, Yang, Li. "On the Continuity of Rotation Representations in Neural Networks." CVPR 2019 (venue per CVF open-access/BibBase listing found in search).
URL: https://arxiv.org/abs/1812.07035 [verified: fetched URL]
Summary: Proves that quaternions and Euler angles are discontinuous representations of SO(3) for networks, while 5D/6D representations are continuous. Discontinuous outputs give much larger errors.
Implication for SignNMFlow: The transformer outputs raw quaternions, which are renormalised per joint, as x1-predictions. This is a discontinuous output map. A 6D-to-rotation head, with the path still on SO(3)/S^3, is a standard improvement.

### D9. Geist, Frey, Zhobro, Levina, Martius. "Learning with 3D rotations, a hitchhiker's guide to SO(3)." ICML 2024.
URL: https://arxiv.org/abs/2404.11735 (details from https://arxiv.org/html/2404.11735v2) [verified: fetched URL]
Summary: A survey and experiments on rotation representations for learning. Quaternions with a half-space (hemisphere) map are fine only for small rotations. Near the boundary "two vectors close to the boundary are far in R but close in SO(3)". "Changing the loss does not fix discontinuities", and distance-picking min(||q-q̂||, ||q+q̂||) actually hurt performance. For general rotations it recommends 6D (Gram-Schmidt) or 9D (SVD).
Implication for SignNMFlow: The hemisphere source plus L2-on-quaternion x1 loss carries the double-cover problem. For joints with large rotations (shoulders, wrists in signing), use a 6D/9D output head or a chordal SO(3) loss (||R - R̂||_F^2). Do not rely on a sign-invariant loss alone.

### D10. Low, Symeonidis-Herzig, Ivashechkin, Mercanoglu Sincan, Bowden. "SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning." ECCV 2026.
URL: https://arxiv.org/abs/2603.10446 [verified: fetched URL]
Summary: Mines sparse keyframes with a sign-segmentation model and uses conditional flow matching to generate dense 3D signing from the keyframe anchors. Reports few-step synthesis and covers four sign languages.
Implication for SignNMFlow: This is the in-domain FM competitor, and SignNMFlow lists it as a baseline. Keyframe anchors are a form of data-dependent coupling/prior (B10, D5), which suggests that the gap between them may come from the prior and not from the path.

---

## Synthesis (10 bullets)

1. **Novelty overlap is substantial.** MFM (B1, NeurIPS 2024) already does "straight path + t(1-t) learned network correction, trained by minimising kinetic energy under a data-induced metric from kernel-weighted local variance (LAND) or k-means-centred RBFs". SignNMFlow's distinct contributions reduce to (i) a Riemannian base path (Exp/Log on (S^3)^J), (ii) a full, prototype-based local-scatter metric, and (iii) the SLP application. GSBM (B5), stochastic interpolants (B7), Shaul et al. (B2) and EGGFM/GeodesicNVS (B8, B9) are further prior work on learned or data-dependent paths. MFM must be cited and compared against (Euclidean MFM on 6D or quaternions).
2. **Check the metric direction.** In MFM the cost metric is the *inverse* of local variance, so moving along the data spread is cheap. SignNMFlow's E[xdot^T G^{-1} xdot] with G = m(S+rho I)^{-1} gives xdot^T (S+rho I) xdot / m, which penalises high-variance directions. The principled choices are Mahalanobis-type xdot^T (S+rho I)^{-1} xdot (MFM/LAND-style) or a state cost of the form -log density (GSBM). Verify the implementation and run both directions.
3. **Conditional-path energy is not marginal-path energy.** Minimising the energy of conditional paths (MFM/SignNMFlow) does not ensure simpler marginal velocity fields (B2). Minibatch Riemannian OT couplings (B3, A7, as in FoldFlow-OT) are the cheaper, proven route to straighter marginal flows and should be both a baseline and a complement.
4. **Isolate the path contribution.** RMG (A11) and PoseD-Flow (A10) show that plain geodesic RFM on rotation manifolds is already strong for human motion, and Proteina (A9) shows that a Euclidean representation plus scale can beat frame/manifold models. Required ablations: Euclidean CFM vs geodesic-only RFM vs RFM + deviation, holding architecture, steps and tricks fixed.
5. **Path change ≈ schedule change.** VDM (C3), Kingma & Gao (C4) and SD3 (C6) show that time parameterisation and weighting are interchangeable. Some gains from the learned deviation may be reproducible with logit-normal or shifted t sampling (C5, C6) or an exponential rotation schedule at inference (A6). Test these first, since they are one-line changes.
6. **Keep x1-prediction, change the loss space.** JiT (C1) and FrameFlow (A6) support predicting clean data. JiT shows that x-pred with v-loss (1/(1-t)^2 weighting, clipped) beats x-pred with x-loss. On the manifold the analogue is the tangent loss ||Log_{x_t}(x1) - Log_{x_t}(x̂1)||^2 / (1-t)^2, clipped, as in FrameFlow.
7. **Fix the rotation output.** Raw-quaternion outputs with per-joint renormalisation and a hemisphere convention are discontinuous (D8, D9). Use a 6D or 9D output head mapped to SO(3), or a chordal or geodesic SO(3) loss. Sign-invariant (distance-picking) quaternion losses were reported to hurt (D9).
8. **Add the standard conditional-FM tricks the method lacks:** CFG with about 10% text dropout (D1, D2; used by Proteina and JiT), self-conditioning on the previous x̂1 (D3; FrameDiff, FrameFlow and Proteina use it, and it is almost free given the re-planning sampler), and EMA or post-hoc EMA (D4, C1).
9. **Replace the uninformative source.** A uniform hemisphere p0 is far from natural poses, and a data-manifold metric is not meaningful there. Data-dependent sources and couplings (PriorGrad D5, Albergo et al. B10), such as an IGSO(3) or wrapped normal around the Fréchet-mean pose (A4) or a text- or keyframe-conditioned coarse pose (as SignSparK's anchors do, D10), shorten paths and help 5-step sampling.
10. **Few-step regime.** If 5 NFE is a hard constraint, Riemannian adaptations of MeanFlow (D6) or shortcut models (D7), or progressive distillation (C2) from a many-step teacher, have the best evidence. The learned-deviation path should be evaluated at matched NFE against these, because path curvature matters less once the model is trained to take large steps.
