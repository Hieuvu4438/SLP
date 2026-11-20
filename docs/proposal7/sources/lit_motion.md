# Literature review: rotation representations, losses, kinematic chains and temporal modelling, and what they mean for SignNMFlow

Compiled 2026-11-19. Scope: SignNMFlow (arXiv 2609.33339). Each frame holds 41 SMPL-X local joint rotations as unit quaternions under a w>=0 convention, plus 10-D expression. Training uses a plain per-joint squared error on quaternions. Evaluation uses DTW-JPE / DTW-PA-JPE after FK, plus back-translation BLEU. A static mean pose scores about as well as the model on DTW-JPE (body).

**Verification policy.**
- Every entry below was confirmed by fetching the listed URL during this session. Title, authors and year come from that page.
- Where the venue came from a different source than the fetched page, the entry says so ("venue via ...").
- Claims marked "(full text)" were checked against the paper's PDF/HTML text, not just the abstract.
- Papers I could not confirm by fetch were dropped. The dropped list is at the end.

---

## A. Rotation representations and losses for learning

### A1. Zhou, Barnes, Lu, Yang, Li. "On the Continuity of Rotation Representations in Neural Networks." CVPR 2019.
- **URLs:** https://arxiv.org/abs/1812.07035 ; venue: https://mlanthology.org/cvpr/2019/zhou2019cvpr-continuity/ (DOI 10.1109/CVPR.2019.00589)
- **Status:** [verified: fetched arXiv abs + mlanthology]
- **Summary:**
  - Defines when a representation is continuous for a neural network.
  - Proves that every 3D rotation representation in 4 or fewer real dimensions is discontinuous, which includes quaternions and Euler angles.
  - Shows that continuous 5D and 6D representations exist.
  - Continuous representations win on an autoencoder sanity test, point-cloud pose estimation and human-pose inverse kinematics.
- **Implication for SignNMFlow:**
  - The w>=0 hemisphere convention is a concrete instance of the discontinuity Zhou describes. Rotations near 180 deg (w close to 0) have two targets that are far apart in R^4 (q and -q).
  - First test: how often do targets have |w| < 0.1? Check per joint, especially shoulders, elbows and wrists, which have large ranges.
  - Then train a 6D (Gram-Schmidt) or 9D+SVD output head as an ablation, keeping the quaternion data format for I/O.

### A2. Brégier. "Deep Regression on Manifolds: A 3D Rotation Case Study." 3DV 2021 (oral).
- **URL:** https://arxiv.org/abs/2103.16317
- **Status:** [verified: fetched arXiv abs; quaternion discussion checked in full text]
- **Summary:**
  - Proposes desirable properties for differentiable maps from R^n onto a manifold: surjectivity, connected and convex pre-images, and local linearity.
  - The normalised-quaternion map x/||x|| fails the pre-image connectivity property (full text).
  - For antipodal quaternions the paper explicitly recommends the loss min||q +/- q*||^2 "in order to avoid border effects of the representation" (full text).
  - Procrustes (SVD) orthonormalisation is generally best. Rotation-vector maps suit small-angle regimes.
- **Implication:**
  - The cheapest change to test is replacing ||q - q*||^2 with min(||q - q*||^2, ||q + q*||^2), or with 1 - <q,q*>^2.
  - It costs a single line and removes the penalty when the model predicts the "other" hemisphere near w = 0.
  - Second ablation: a Procrustes/9D head.

### A3. Levinson, Esteves, Chen, Snavely, Kanazawa, Rostamizadeh, Makadia. "An Analysis of SVD for Deep Rotation Estimation." NeurIPS 2020.
- **URLs:** https://arxiv.org/abs/2006.14616 ; venue: https://proceedings.neurips.cc/paper/2020/hash/fec3392b0dc073244d38eba1feb8e6b7-Abstract.html
- **Status:** [verified: fetched arXiv abs; venue via NeurIPS proceedings listing in search]
- **Summary:**
  - Argues theoretically that symmetric SVD orthogonalisation is the natural projection onto SO(3).
  - Shows empirically that swapping it in for quaternions, Euler angles or axis-angle gives state-of-the-art results across many supervised and unsupervised tasks, including pose estimation and IK learning.
- **Implication:** This is the strongest single "drop-in" alternative output head. Predict 9 numbers per joint, project with SVD, and compute the loss on rotation matrices (chordal) or on FK positions.

### A4. Geist, Frey, Zhobro, Levina, Martius. "Learning with 3D rotations, a hitchhiker's guide to SO(3)." ICML 2024.
- **URL:** https://arxiv.org/abs/2404.11735 (HTML full text read)
- **Status:** [verified: fetched arXiv abs + HTML]
- **Summary:**
  - A survey and practical guide.
  - For rotation estimation (network outputs) it recommends R^9+SVD or R^6+Gram-Schmidt.
  - Representations with 4 or fewer dimensions must be discontinuous.
  - Quaternions with a half-space map (exactly the w>=0 convention) are "a good option" only if targets are small rotations.
  - Sign-invariant losses such as min(||q1-q2||, ||q1+q2||) or 1-|q1·q2| help, but "changing the loss does not fix discontinuities".
- **Implication:**
  - SignNMFlow's setup (quaternion, half-space map, raw L2) is the configuration this guide flags as acceptable only for small angles.
  - Finger joints are mostly small rotations. Shoulder, collar and elbow joints are not.
  - Test a mixed scheme: keep quaternions for fingers and use 6D/9D for upper-body joints. Alternatively use 6D everywhere.

### A5. Peretroukhin, Giamou, Rosen, Greene, Roy, Kelly. "A Smooth Representation of Belief over SO(3) for Deep Rotation Learning with Uncertainty." RSS 2020.
- **URL:** https://arxiv.org/abs/2006.01031
- **Status:** [verified: fetched arXiv abs]
- **Summary:**
  - Represents a rotation by a 4x4 symmetric matrix A. The output quaternion is the eigenvector with the smallest eigenvalue, so it is sign-agnostic by construction.
  - The representation is smooth, helps with large rotations, and encodes a Bingham belief over quaternions, which allows uncertainty estimates.
- **Implication:**
  - This is a sign-invariant quaternion head that stays compatible with SignNMFlow's quaternion data.
  - The Bingham view also gives a principled per-joint uncertainty, useful for diagnosing which joints the model "gives up" on and regresses to the mean.

### A6. Dünkel, Salzmann, Pfaff. "Normalizing Flows on the Product Space of SO(3) Manifolds for Probabilistic Human Pose Modeling" (HuProSO3). CVPR 2024.
- **URL:** https://arxiv.org/abs/2404.05675
- **Status:** [verified: fetched arXiv abs]
- **Summary:**
  - A normalizing flow defined directly on SO(3)^J for human joint rotations.
  - Gives exact likelihoods and beats existing methods in three pose-modelling applications.
- **Implication:**
  - SignNMFlow models Euclidean quaternion vectors, so its density sits in R^4 per joint, not on SO(3).
  - A manifold-aware flow, or at minimum a tangent-space parameterisation, avoids putting probability mass off-manifold and avoids the q/-q split.
  - This is a larger change. Use it as a reference design if the representation ablations (A1–A5) show large gains.

### A7. Pavllo, Grangier, Auli. "QuaterNet: A Quaternion-based Recurrent Model for Human Motion." BMVC 2018. Extended as Pavllo, Feichtenhofer, Auli, Grangier, "Modeling Human Motion with Quaternion-based Neural Networks," IJCV 2019.
- **URLs:** https://arxiv.org/abs/1805.06485 ; https://arxiv.org/abs/1901.07677
- **Status:** [verified: fetched both arXiv abs]
- **Summary:**
  - The closest precedent to SignNMFlow: joint rotations as quaternions.
  - The loss "performs forward kinematics on a skeleton to penalize absolute position errors instead of angle errors."
  - The IJCV abstract states the motivating problem directly: rotation regression "can accumulate errors along the kinematic chain".
- **Implication:**
  - Direct evidence that quaternion models should not be trained with a quaternion-space loss alone.
  - Add a differentiable SMPL-X FK position loss on the same joints the metric uses: body positions, plus wrist-relative hand positions. Keep the quaternion loss as a regulariser.

---

## B. Motion-generation representations, losses and architectures

### B1. Guo, Zou, Zuo, Wang, Ji, Li, Cheng. "Generating Diverse and Natural 3D Human Motions From Text" (HumanML3D). CVPR 2022, pp. 5152–5161.
- **URL:** https://github.com/EricGuo5513/HumanML3D (official repo with BibTeX; CVF page returned 403)
- **Status:** [verified: fetched official repo]
- **Summary:**
  - Introduces the HumanML3D dataset.
  - Introduces the redundant per-frame feature vector derived from 3D joints. The repo describes "rotation invariant" and rotation features with Mean/Std normalisation. The 7-group breakdown (root angular and linear velocity, root height, local joint positions, joint velocities, 6D rotations, foot contacts) is taken from the B3 full text.
  - It became the de facto text-to-motion representation.
- **Implication:**
  - SignNMFlow uses only local rotations. HumanML3D-style redundancy adds root-relative positions and per-joint velocities next to rotations, so the training loss directly "sees" the evaluated quantities.
  - Test adding FK-derived positions as auxiliary output channels, or as a loss.

### B2. Tevet, Raab, Gordon, Shafir, Cohen-Or, Bermano. "Human Motion Diffusion Model" (MDM). ICLR 2023.
- **URLs:** https://arxiv.org/abs/2209.14916 ; venue BibTeX: https://github.com/GuyTevet/motion-diffusion-model
- **Status:** [verified: fetched arXiv abs + repo]
- **Summary:**
  - Transformer diffusion that predicts the clean sample rather than the noise.
  - The authors state that this "enables the use of established geometric losses on the locations and velocities of the motion", for example foot contact.
- **Implication:**
  - A flow/diffusion model can carry geometric losses as long as it exposes an x0 estimate. In flow matching that is x_t + (1-t)·v.
  - Apply FK-position and velocity losses on the predicted x0, possibly restricted to small t or weighted by t.

### B3. Meng, Xie, Peng, Han, Jiang. "Rethinking Diffusion for Text-Driven Human Motion Generation: Redundant Representations, Evaluation, and Masked Autoregression." CVPR 2025.
- **URL:** https://arxiv.org/abs/2411.16575 (HTML full text read)
- **Status:** [verified: fetched arXiv abs + HTML]
- **Summary (full text):**
  - The redundant 263-D HumanML3D vector helps VQ models but hurts diffusion.
  - Mixed feature types do not z-normalise to a common Gaussian.
  - Training on only the "essential" continuous groups (root and local positions) plus x0/velocity prediction cuts FID substantially. Without the reformation, FID rises from 0.116 to 2.196. ε-prediction MDM reaches FID 1.574–31.3 versus 0.518 for x0-prediction.
  - The standard evaluator is dominated by redundant dimensions.
- **Implication:**
  - Two lessons carry over to SignNMFlow.
  - First, per-feature normalisation of quaternion components versus expression components should be checked. Distributions differ greatly between the jaw, fingers and shoulders.
  - Second, the paper shows that evaluators/metrics can be gamed by particular dimensions. That parallels the mean-pose problem with DTW-JPE (body).

### B4. Meng, Han, Peng, Xie, Jiang. "Absolute Coordinates Make Motion Generation Easy" (ACMDM). arXiv preprint 2025.
- **URL:** https://arxiv.org/abs/2505.19377
- **Status:** [verified: fetched arXiv abs; no venue found]
- **Summary:**
  - Replaces the pelvis- and previous-frame-relative representation with absolute global joint coordinates.
  - Reports significantly higher fidelity and text alignment with a plain Transformer and "without auxiliary kinematic losses".
  - Shows good scaling and extends to SMPL-H mesh vertices.
- **Implication:**
  - Supports generating positions (or positions jointly with rotations) rather than local rotations only.
  - For sign language this suggests an ablation that generates root-relative 3D joint positions, with wrist-relative positions for the hands.
  - Rotations would then be recovered by IK, or by a light rotation head, only for avatar rendering.
  - Caveat: preprint only.

### B5. Zou, Yuan, Du, Wang, Liu, Xu, Chen, Ji. "ParCo: Part-Coordinating Text-to-Motion Synthesis." ECCV 2024.
- **URL:** https://arxiv.org/abs/2403.18512
- **Status:** [verified: fetched arXiv abs]
- **Summary:**
  - Discretises whole-body motion into part motions, each generated by a lightweight generator.
  - A part-coordination module links the generators.
  - Improves HumanML3D/KIT-ML results at low compute.
- **Implication:**
  - Hands (30 joints) and the upper body (10 joints) have very different statistics and informational roles in signing.
  - Separate part streams with coordination, e.g. body / left hand / right hand / face, are a well-supported design.

### B6. Yi, Liang, Liu, Cao, Wen, Bolkart, Tao, Black. "Generating Holistic 3D Human Motion from Speech" (TalkSHOW). CVPR 2023.
- **URL:** https://arxiv.org/abs/2212.04420
- **Status:** [verified: fetched arXiv abs]
- **Summary:**
  - Generates SMPL-X body, hands and face from speech.
  - The face is modelled separately because it tracks audio closely.
  - Body and hands use a compositional VQ-VAE with a cross-conditional autoregressive model.
- **Implication:**
  - Precedent for treating SMPL-X hands as their own (but cross-conditioned) stream.
  - Treating the 10-D expression and jaw separately from manual articulators is also supported.

### B7. Liu, Zhu, Becherini, Peng, Su, Zhou, Zhe, Iwamoto, Zheng, Black. "EMAGE: Towards Unified Holistic Co-Speech Gesture Generation via Expressive Masked Audio Gesture Modeling" (introduces the BEAT2 dataset). CVPR 2024.
- **URL:** https://arxiv.org/abs/2401.00374 (HTML full text read)
- **Status:** [verified: fetched arXiv abs ("CVPR Camera Ready") + HTML]
- **Summary (full text):**
  - SMPL-X/FLAME holistic gestures with joints in Rot6D.
  - Four separate VQ-VAEs: face, upper body, hands (180-D) and lower body.
  - VQ-VAE reconstruction, velocity and acceleration terms use geodesic and L1 losses.
  - A global-motion predictor uses foot-contact labels.
- **Implication:**
  - The closest SMPL-X hand-generation recipe to SignNMFlow, with three differences from it: (i) 6D not quaternion, (ii) a geodesic loss, (iii) explicit velocity and acceleration losses.
  - All three are cheap, directly testable changes.

### B8. Chen, Liu, Wang, Zeng, Li, Chen. "DiffSHEG: A Diffusion-Based Approach for Real-Time Speech-driven Holistic 3D Expression and Gesture Generation." CVPR 2024.
- **URL:** https://arxiv.org/abs/2401.04747
- **Status:** [verified: fetched arXiv abs]
- **Summary:**
  - Joint diffusion over expressions and gestures, with uni-directional expression-to-gesture information flow in the transformer.
  - Uses outpainting-style sampling for arbitrary-length sequences.
- **Implication:**
  - For sign language, non-manual signals and manual signals are coupled.
  - DiffSHEG's directed conditioning is one way to model the coupling explicitly instead of concatenating 10-D expression to 164-D quaternions under one loss.

### B9. Chen, Jiang, Liu, Huang, Fu, Chen, Yu, Yu. "Executing your Commands via Motion Diffusion in Latent Space" (MLD). CVPR 2023.
- **URLs:** https://arxiv.org/abs/2212.04048 ; venue: https://github.com/ChenFengYe/motion-latent-diffusion
- **Status:** [verified: fetched arXiv abs + repo BibTeX]
- **Summary:**
  - A VAE compresses motion sequences into a latent, and diffusion runs in that latent.
  - Better quality, and two orders of magnitude faster than raw-motion diffusion.
- **Implication:**
  - A motion autoencoder can be trained with FK, velocity and sign-invariant losses once.
  - The generator then models a smooth latent, which isolates representation problems from generative-model problems.

### B10. Guo, Mu, Javed, Wang, Cheng. "MoMask: Generative Masked Modeling of 3D Human Motions." CVPR 2024.
- **URLs:** https://arxiv.org/abs/2312.00063 ; venue: https://ericguo5513.github.io/momask/
- **Status:** [verified: fetched arXiv abs + project page]
- **Summary:**
  - Residual VQ tokenisation: a base layer plus residual layers, up to 5 residual layers in the project page comparisons.
  - A masked transformer predicts base tokens and a residual transformer predicts the finer layers.
  - FID 0.045 on HumanML3D.
- **Implication:**
  - Discrete or RVQ tokens are a structural antidote to regression-to-the-mean, because outputs are always decoded from codebook entries.
  - A useful baseline family for SLP, alongside SOKE (D3).

### B11. Ng, Ginosar, Darrell, Joo. "Body2Hands: Learning to Infer 3D Hands from Conversational Gesture Body Dynamics." CVPR 2021.
- **URL:** https://arxiv.org/abs/2007.12287 (venue via CVF open-access listing in search)
- **Status:** [verified: fetched arXiv abs]
- **Summary:**
  - Predicts 3D hand pose over time from arm and body motion alone.
  - Motivated by the strong correlation between body motion and hand gestures.
  - Outperforms prior hand-synthesis methods and works as a prior for monocular hand estimation.
- **Implication:**
  - Hands should be conditioned on the (generated) arm and wrist trajectory.
  - A two-stage or cross-conditioned design (body then hands) may help.
  - Hand metrics that are wrist-aligned test exactly this conditional quality.

### B12. Pavlakos, Choutas, Ghorbani, Bolkart, Osman, Tzionas, Black. "Expressive Body Capture: 3D Hands, Face, and Body from a Single Image" (SMPL-X). CVPR 2019.
- **URL:** https://arxiv.org/abs/1904.05866
- **Status:** [verified: fetched arXiv abs]
- **Summary:**
  - Introduces SMPL-X, a body model with articulated hands and an expressive face.
  - Introduces SMPLify-X fitting, which includes a learned neural pose prior trained on MoCap.
- **Implication:**
  - SignNMFlow's evaluation goes through SMPL-X FK, so the same differentiable SMPL-X layer can be used in training.
  - A learned pose prior can act as a plausibility regulariser for fingers.
  - Note: the abstract does not name VPoser.

---

## C. Kinematic-chain-aware errors and losses

### C1. Aksan, Kaufmann, Hilliges. "Structured Prediction Helps 3D Human Motion Modelling" (SPL). ICCV 2019.
- **URL:** https://arxiv.org/abs/1910.09070
- **Status:** [verified: fetched arXiv abs]
- **Summary:**
  - A structured prediction layer made of small per-joint networks, connected like the kinematic chain.
  - Each joint's prediction is conditioned on its parent.
  - The loss is decomposed per joint.
  - Improves forecasting across base networks and angle representations.
- **Implication:**
  - SignNMFlow predicts all 41 rotations in parallel from a shared feature.
  - A cheap structured output head could reduce inconsistent finger/wrist configurations: shoulder, then elbow, then wrist, then finger chains, each conditioned on its parent's output.

### C2. Wang, Zhang. "3D Human Mesh Recovery with Sequentially Global Rotation Estimation" (SGRE). ICCV 2023.
- **URL:** https://github.com/kennethwdk/sgre (official repo confirming title, authors, ICCV 2023; DOI 10.1109/ICCV51070.2023.01373 via search)
- **Status:** [verified: fetched official repo]
- **Summary:**
  - Predicts each joint's global rotation sequentially along the chain instead of local rotations.
  - The authors report this alleviates the error accumulation along the kinematic chain that relative-rotation regression causes.
  - Gains are largest at end joints.
- **Implication:**
  - Fingertips and wrists are end joints of long chains: pelvis/spine, then collar, shoulder, elbow, wrist, then three finger links.
  - Test a loss on global (accumulated) rotations, R_global = Π R_local, at the wrist and the finger roots. Alternatively predict wrist orientation globally.

### C3. Liu, Wu, Jin, Ji, Liu, Lu, Cheng. "Investigating Pose Representations and Motion Contexts Modeling for 3D Motion Prediction" (AHMR). IEEE TPAMI 2021/22.
- **URL:** https://arxiv.org/abs/2112.15012
- **Status:** [verified: fetched arXiv abs; TPAMI acceptance and DOI 10.1109/TPAMI.2021.3139918 in arXiv comments]
- **Summary:**
  - Systematically studies how the choice of pose representation affects motion prediction and finds it has "a critical impact".
  - Replaces the common L2 loss with geodesic and forward-kinematics losses, citing their geometric meaning.
- **Implication:**
  - Independent support for adding a geodesic rotation loss and an FK loss to SignNMFlow's plain L2.
  - The abstract gives no numbers. Read the full text before quoting effect sizes.

### C4. (Grey literature) theorangeduck.com. "Joint Error Propagation." Blog post, 17 Aug 2025. Author not named on the page; the site is that of the "orangeduck" GitHub account.
- **URL:** https://www.theorangeduck.com/page/joint-error-propagation
- **Status:** [verified: fetched page; NOT peer-reviewed]
- **Summary:**
  - Makes the lever-arm argument explicitly: a small local rotation error near the root moves the whole descendant subtree, so root-side joints need higher accuracy.
  - Compares an FK loss (effective, but reported as about 100x slower to train, with poor gradients) against cheap per-joint weights.
  - Three weighting schemes: discounted descendant bone length, cylinder surface area, and skinned-mesh area × distance. The skinned-mesh scheme is reported best.
- **Implication:**
  - A zero-cost first experiment: weight each joint's quaternion loss by a descendant-length or skinned-vertex lever arm computed from the SMPL-X template.
  - Compare against a full SMPL-X FK loss.
  - Treat this as practitioner evidence only.

---

## D. Sign-language-production specific

### D1. Saunders, Camgoz, Bowden. "Progressive Transformers for End-to-End Sign Language Production." ECCV 2020.
- **URLs:** https://arxiv.org/abs/2004.14874 ; venue via ECVA PDF https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560664.pdf (search listing; DOI 10.1007/978-3-030-58621-8_40)
- **Status:** [verified: fetched arXiv abs]
- **Summary:**
  - First text-to-continuous-3D-skeleton SLP transformer, using a counter decoding mechanism.
  - Uses data augmentation against drift.
  - Introduces back-translation evaluation on PHOENIX14T.
- **Implication:**
  - The origin of the back-translation protocol SignNMFlow reports.
  - The paper regresses skeleton positions with MSE, which is a known regression-to-mean setting (see D2).

### D2. Saunders, Camgoz, Bowden. "Adversarial Training for Multi-Channel Sign Language Production." BMVC 2020.
- **URLs:** https://arxiv.org/abs/2008.12405 ; venue via BMVA archive https://www.bmva-archive.org.uk/bmvc/2020/assets/papers/0223.pdf (search listing)
- **Status:** [verified: fetched arXiv abs]
- **Summary:**
  - Explicitly attributes the "under-articulated output" of prior SLP to "regression to the mean".
  - Adds a text-conditioned adversarial discriminator, plus non-manual (face/mouthing) channels.
  - State-of-the-art back-translation on PHOENIX14T at the time.
- **Implication:**
  - Direct precedent for SignNMFlow's observation that the mean pose is about as good as the model.
  - SignNMFlow is already a generative (flow) model. If samples still collapse toward the mean, check whether the loss or sampler effectively behaves like a regressor. Examples: low-temperature sampling, or heavy guidance.
  - Report articulation statistics (hand motion energy, per-joint velocity variance vs ground truth).

### D3. Zuo, Potamias, Ververas, Deng, Zafeiriou. "Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator" (SOKE). ICCV 2025.
- **URL:** https://arxiv.org/abs/2411.17799
- **Status:** [verified: fetched arXiv abs]
- **Summary:**
  - Autoregressive LM-based 3D sign-avatar generation.
  - A decoupled tokenizer discretises signs into per-body-part token sequences.
  - Multi-head decoding.
  - Retrieval of dictionary signs as auxiliary conditions.
- **Implication:**
  - A strong, recent SLP comparison point that uses part-decoupled discrete tokens.
  - Supports a body/hand split (B5–B7) in the sign domain specifically.

### D4. Baltatzis, Potamias, Ververas, Sun, Deng, Zafeiriou. "Neural Sign Actors: A diffusion model for 3D sign language production from text." CVPR 2024.
- **URL:** https://arxiv.org/abs/2312.02702
- **Status:** [verified: fetched arXiv abs]
- **Summary:**
  - Diffusion SLP on 4D SMPL-X signing avatars.
  - The denoiser is an "anatomically informed" graph network defined on the SMPL-X skeleton.
  - Outperforms prior SLP methods.
- **Implication:**
  - Evidence that skeleton-topology-aware denoisers help for SMPL-X sign generation.
  - Related to C1: replace or augment SignNMFlow's flat token-per-frame network with joint-graph structure.

### D5. Tang, He, Guo, Wei, Li, Hong. "Sign-IDD: Iconicity Disentangled Diffusion for Sign Language Production." AAAI 2025.
- **URL:** https://arxiv.org/abs/2412.13609
- **Status:** [verified: fetched arXiv abs]
- **Summary:**
  - Argues that treating joints as independent coordinates ignores relative structure.
  - Converts poses into a bone representation, a 3D direction plus a 1D length per adjacent joint pair.
  - Disentangles direction and length in an attribute-controllable diffusion model. Evaluated on PHOENIX14T and USTC-CSL.
- **Implication:**
  - Bone directions are the position-space analogue of local rotations and are lever-arm aware by construction.
  - A bone-direction loss is cheap: compute it from FK joints and compare with cosine similarity per bone.
  - It sits between the pure quaternion loss and a full FK position loss.

### D6. Shalev-Arkushin, Moryossef, Fried. "Ham2Pose: Animating Sign Language Notation into Pose Sequences." CVPR 2023, pp. 21046–21056.
- **URLs:** https://arxiv.org/abs/2211.13613 ; venue via CVF open-access listing (search)
- **Status:** [verified: fetched arXiv abs]
- **Summary:**
  - Generates poses from HamNoSys with transformer encoders under weak supervision.
  - Proposes nDTW, a DTW distance over normalised keypoint trajectories that accounts for missing keypoints.
  - Validates it on AUTSL as a better pose-sequence distance than existing measures.
- **Implication:**
  - SignNMFlow's DTW-JPE is in this family.
  - Ham2Pose's metric-validation protocol is reusable: does the metric rank same-sign pairs closer than different-sign pairs?
  - Use it to show that DTW-JPE (body) fails to separate a mean pose from real signing, and to justify adding hand-focused and dynamics-aware metrics.

### D7. Ginosar, Bar, Kohavi, Chan, Owens, Malik. "Learning Individual Styles of Conversational Gesture." CVPR 2019.
- **URL:** https://arxiv.org/abs/1906.04160 (full text read)
- **Status:** [verified: fetched arXiv abs + PDF text]
- **Summary (full text):**
  - Speech-to-gesture with L1 regression on keypoint stacks.
  - Because gesture is multimodal, they add an adversarial discriminator "to avoid regressing to the mean of all possible modes of gesture".
  - Large temporal context gives smoothness.
- **Implication:**
  - Canonical evidence that a pointwise regression loss under a one-to-many mapping (text to sign realisation) produces averaged motion.
  - Any regression-style auxiliary loss added to SignNMFlow (FK, velocity) should be applied to the flow's x0 estimate. It should be checked for mean-collapse, e.g. by comparing motion energy against ground truth.

---

## E. Temporal modelling, smoothness and baselines

### E1. Martinez, Black, Romero. "On human motion prediction using recurrent neural networks." CVPR 2017.
- **URL:** https://arxiv.org/abs/1705.02445
- **Status:** [verified: fetched arXiv abs]
- **Summary:**
  - Shows that "a simple baseline that does not attempt to model motion at all" matches the state of the art under the standard evaluation.
  - Re-examines evaluation, losses and training procedures.
- **Implication:**
  - The motion-prediction analogue of SignNMFlow's mean-pose result.
  - Precedent for always reporting trivial baselines next to the model: static mean pose, per-sentence mean, nearest-neighbour retrieval.
  - Also precedent for treating metrics that do not separate model from baseline as uninformative.

### E2. Mao, Liu, Salzmann, Li. "Learning Trajectory Dependencies for Human Motion Prediction." ICCV 2019 (oral).
- **URL:** https://arxiv.org/abs/1908.05436 (full text read)
- **Status:** [verified: fetched arXiv abs + PDF text]
- **Summary (full text):**
  - Encodes each joint trajectory with DCT coefficients and predicts in that "trajectory space".
  - "Truncating some of the high frequencies can prevent generating jittery motion."
  - Angle representations contain large jumps (discontinuities), which make low-coefficient DCT reconstruction lossy. 3D coordinates do not have this problem.
- **Implication:**
  - Two direct lessons.
  - (i) A DCT/low-pass temporal parameterisation is a principled jitter control.
  - (ii) Rotation targets with jumps damage temporal modelling. The w>=0 sign convention creates exactly such jumps whenever a joint crosses w = 0 between frames.
  - Measure per-joint sign-flip rates in the training targets. Consider enforcing temporal sign continuity (flip q_t if <q_t, q_{t-1}> < 0) within sequences instead of a global w>=0.

### E3. Zhang, Black, Tang. "We are More than Our Joints: Predicting how 3D Bodies Move" (MOJO). CVPR 2021.
- **URL:** https://arxiv.org/abs/2012.00619 (full text read)
- **Status:** [verified: fetched arXiv abs ("camera ready, cvpr") + PDF text]
- **Summary (full text):**
  - A CVAE with a latent DCT space.
  - Sampling latent frequencies reintroduces high-frequency detail.
  - The loss combines frame-wise reconstruction and frame-wise velocity reconstruction ("the velocity reconstruction term can further improve temporal smoothness").
  - SMPL-X is fit to predicted markers.
  - On the velocity-loss weight α: "a larger value causes over-smooth and loses motion details, whereas a smaller value can cause jitters."
- **Implication:**
  - Direct evidence of the jitter vs over-smoothing trade-off for a velocity loss.
  - If SignNMFlow adds a velocity loss, sweep its weight.
  - Monitor both jitter (acceleration/jerk magnitude vs GT) and articulation (velocity variance vs GT). Do not monitor DTW-JPE alone, since over-smoothing pushes toward the mean pose, which the metric does not punish.

### E4. Zeng, Yang, Ju, Li, Wang, Xu. "SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos." ECCV 2022.
- **URL:** https://arxiv.org/abs/2112.13715
- **Status:** [verified: fetched arXiv abs]
- **Summary:**
  - A temporal-only, per-joint, fully connected refinement network over long windows.
  - Plugs onto any pose estimator, reduces jitter, and improves accuracy on hard frames, across 5 datasets and 11 backbones.
- **Implication:**
  - A cheap post-hoc option if SignNMFlow samples are jittery, especially the fingers.
  - It must be evaluated with the same over-smoothing guardrails as E3.
  - Also a source for jitter/acceleration-error metrics.

---

## Dropped / not verified (not cited above)
- **Huang, Pan, Zhao, Tian, "Towards Fast and High-Quality Sign Language Production" (ACM MM 2021; origin of DTW-MJE):** citation seen only in search snippets. The paper page was not fetched, so it is dropped.
- **DeepPhase / periodic autoencoders (Starke et al.):** not checked.
- **HybrIK (CVPR 2021):** fetched, but the abstract makes no kinematic-chain error-accumulation claim, so it is not relevant enough to include.
- **BEAT (ECCV 2022), MotionLCM (ECCV 2024), MANO/Embodied Hands (SIGGRAPH Asia 2017), HKMR (ECCV 2020), SignAvatars (ECCV 2024):** all verified by fetching arXiv abs (2203.05297, 2404.19759, 2201.02610, 2003.04232, 2310.20436). Omitted from the main list for length, since they add little beyond EMAGE, MLD, B12, C1/C2 and D3/D4. They can be added on request.
- **MOJO supplement page on the CVF site:** returned 403. The trade-off quote was instead taken from the arXiv full text.

---

## Synthesis: 10 evidence-backed improvement directions for SignNMFlow

1. **Make the loss sign-invariant (cheapest fix).** Replace ||q - q*||^2 with min(||q - q*||^2, ||q + q*||^2), or with 1 - <q,q*>^2. Brégier (A2) recommends this explicitly. Geist et al. (A4) note it does not remove output discontinuity, so pair it with item 2.

2. **Ablate the output representation.** Compare quaternion with w>=0, 6D Gram-Schmidt, and 9D+SVD (A1, A3, A4). Optionally keep quaternions for small-angle finger joints and use 6D/9D for the shoulder, collar and elbow, following Geist's small-angle guidance. Before running anything, measure how many targets have |w| < 0.1 per joint. That predicts where it matters.

3. **Enforce temporal quaternion continuity in the data.** Within each sequence, choose sign(q_t) to agree with q_{t-1} instead of a global w>=0. Mao et al. (E2) show that discontinuities in angle trajectories damage temporal models. The hemisphere convention injects such jumps whenever a joint crosses w = 0.

4. **Add a differentiable SMPL-X FK position loss on the flow's x0 estimate.** QuaterNet (A7), MDM (B2) and AHMR (C3) all use FK/geometric losses. This aligns training with DTW-JPE. Use separate terms for body joints (root-relative) and for wrist-relative hand joints, since that matches the PA/wrist-aligned hand metric.

5. **Lever-arm / kinematic weighting as a zero-cost precursor to FK.** Weight per-joint rotation losses by descendant chain length or skinned-vertex influence (C4, grey literature). Also consider a global-rotation loss at the wrist and finger roots (SGRE, C2), because end-effectors accumulate parent errors (A7 IJCV, C2).

6. **Add velocity (and possibly acceleration) losses, and sweep their weights.** EMAGE (B7) uses velocity and acceleration terms, MDM (B2) a velocity term, MOJO (E3) a velocity term. MOJO reports the trade-off explicitly: too high over-smooths, too low jitters. Monitor motion energy against GT, not just DTW-JPE.

7. **Fix the evaluation before trusting gains.** A mean-pose baseline matching the model is the Martinez (E1) / Saunders (D2) failure mode. Always report static-mean and retrieval baselines. Add dynamics-sensitive metrics: per-joint velocity variance ratio vs GT, hand articulation energy, jitter/acceleration error, and an FID-style distribution metric. Validate metrics with a Ham2Pose-style test (D6) of whether the metric separates same-sign from different-sign pairs.

8. **Treat regression-to-the-mean as a generative-model problem.** Check whether sampling temperature or guidance collapses outputs (D2, D7). Consider discrete/RVQ or latent tokenisation for hands (B10, D3, B9), which structurally avoids averaging.

9. **Decompose by body part, with cross-conditioning.** Use separate streams or tokenisers for body, left hand, right hand and face/jaw, with coordination. Supported by ParCo (B5), TalkSHOW (B6), EMAGE (B7), SOKE (D3), and DiffSHEG for face-to-gesture conditioning (B8). Condition hands on generated arm/wrist motion, following Body2Hands (B11).

10. **Consider position-space or skeleton-aware generation as a bigger bet.** Options:
    - Absolute or root-relative joint coordinates (ACMDM, B4, preprint).
    - Bone direction plus length (Sign-IDD, D5).
    - Skeleton-graph denoisers (Neural Sign Actors D4, SPL C1).
    - Manifold-native SO(3)^J flows (HuProSO3, A6).

    Try these only after items 1–6 quantify how much is gained by fixing the loss within the current quaternion pipeline.
