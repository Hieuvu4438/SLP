# SignRAE for Text-to-3D Sign Language Production

## 1. Related work

Progressive Transformers generate variable-length signing poses directly from spoken-language sentences [1]. Neural Sign Actors uses an anatomically structured diffusion model to generate body, hand, and facial motion [2]. SOKE learns separate discrete tokenizers for the body and each hand, generates their tokens autoregressively, and can condition generation on retrieved isolated signs [3]. SignRAE uses the SMPL-X fits released with SOKE and generates the full sequence in a continuous motion latent.

Motion Latent Diffusion learns its generative space by reconstructing motion [4]. Joint-embedding predictive architectures learn a representation by predicting the features of hidden inputs [5]. MoRAE applies this principle to human motion by masking entire joint trajectories, then compresses the frozen encoder's features before fitting a flow [6]. SignRAE adapts this design to signing by masking at several anatomical scales. When a finger is hidden, the visible fingers provide evidence about the handshape. When the entire hand is hidden, the encoder has to draw on the arm, the torso, and the opposite hand.

## 2. Method

### 2.1 Task and motion representation

Given a written sentence and its target sign language, SignRAE generates a sequence of signing poses whose length it predicts from the sentence. Each frame contains 133 SMPL-X parameters standardized with SOKE's released statistics. The upper body accounts for 30 axis-angle coordinates, each hand for 45, and the face for three jaw rotation coordinates and ten expression coefficients. The generator learns from paired text and fitted motion without gloss annotations or a retrieval dictionary. The motion encoder and decoder are shared across languages. We fit a separate text-conditioned generator for each dataset. [3, 7]

### 2.2 Self-supervised motion encoder

We represent each frame by its pose parameters and their first differences. A temporal Transformer receives one token per frame. We partition the pose into 42 groups, one for each joint rotation and one for all the expression coefficients. For each training clip, we choose to mask individual groups, kinematic chains such as arms and fingers, or units that keep each hand intact. We choose these three granularities with probabilities 0.4, 0.4, and 0.2, then draw whole units until at least 13 groups are hidden. The same groups are masked throughout the clip. Learned mask values replace both the pose channels and their first differences.

The student encodes the masked clip. A temporal predictor maps its features to the teacher's normalized final-layer features from the complete clip. We minimize a smooth L1 loss over all valid frames. The teacher receives an exponential moving average of the student's encoder weights, with momentum increased linearly from 0.996 to 1. Whole-trajectory masking prevents the student from reading the hidden coordinates in adjacent frames. The teacher targets provide supervision for predicting how the visible and hidden joints move together. [5, 6]

### 2.3 Framewise motion latents

We freeze the teacher and use it to encode complete motion clips. At each frame we sum the features from its last four blocks and apply layer normalization. One temporal self-attention layer with a residual connection processes these features, and a linear projection predicts the mean and log variance of a 64-dimensional diagonal Gaussian at each frame. This bottleneck compresses the feature channels while preserving one latent vector per frame. A temporal Transformer decodes the latent sequence directly into the 133 pose parameters. [6]

During autoencoder training, the decoder receives samples from the posterior. We optimize mean squared reconstruction error over the valid frames and pose channels, with a Kullback–Leibler penalty of weight $10^{-4}$ toward a standard Gaussian prior. We select the autoencoder by validation reconstruction error and freeze it before training the generator. The flow uses the posterior means as targets. We standardize each latent channel using its mean and standard deviation over the training frames of the target dataset. [6]

### 2.4 Text-conditioned latent flow

Let $z$ be the standardized latent sequence of a training clip. We draw standard Gaussian noise $\epsilon$ of the same shape and one time $t$ uniformly from $[0,1]$. The path from noise to motion and its velocity are

$$
z_t=(1-t)\epsilon+t z,\qquad \frac{d z_t}{d t}=z-\epsilon.
$$

The generator takes $z_t$ and $t$ together with the sentence and target language, and predicts $z-\epsilon$. We minimize the mean squared prediction error over valid frames and latent channels. A single draw of $t$ applies to the entire clip, so the Transformer sees all frames at the same point along the path. [6, 8]

We initialize the text encoder from the multilingual mBART checkpoint distributed with SOKE and fine-tune it with the generator. The generator projects latent frames to its hidden width and adds sinusoidal frame positions. The sum of the projected mean-pooled text features, a learned sign-language embedding, and the time embedding controls adaptive layer normalization in the DiT blocks. Cross-attention lets each motion frame attend to the contextualized text tokens. During training, we independently mask access to individual encoded text tokens with probability 0.1 and drop text conditioning from the flow for 10% of examples. The sign-language embedding remains available when the text is dropped. [3, 9, 10]

### 2.5 Duration prediction and sampling

A two-layer head predicts the log frame count from the mean-pooled text features and a learned language embedding. Its absolute error against the log reference frame count is added to the flow loss with unit weight. At inference, we exponentiate the predicted log length, round it to a frame count, and constrain it to 40–400 frames. The generator and decoder both operate at this length.

We sample a Gaussian latent sequence at the predicted length. Its standard deviation is set by the sampling temperature. We integrate the text-conditioned velocity with five equally spaced forward Euler steps from $t=0$ to $t=1$, with one model evaluation per step. The selected guidance scale is 1, which uses the conditional velocity directly. We undo the latent standardization and decode the resulting sequence to produce one signing motion per sentence. [11]

## 3. Experimental setup

We use the official splits of Phoenix-2014T, CSL-Daily, and How2Sign with the SMPL-X fits distributed by SOKE. For How2Sign, clips recorded above 24 frames per second are subsampled to 24 frames per second using SOKE's frame selection rule. We apply its exclusions for failed fits. Across datasets, we resample clips outside the 40–400 frame range to the nearest bound and center-crop other clips to a multiple of four frames. [3, 12–14]

For the two motion-only stages we pool 56,429 training clips, comprising 7,092 from Phoenix-2014T, 18,399 from CSL-Daily, and 30,938 from How2Sign. We sample contiguous training windows with a requested length drawn uniformly between 32 and 400 frames, using the full clip when it is shorter than the sampled length. We train the flow and duration head on the preprocessed clips at their full lengths.

| Component | Architecture | Training |
| --- | --- | --- |
| Motion encoder | 8 Transformer layers, width 768, 12 heads | 60,000 updates, batch size 128 |
| Motion predictor | 6 Transformer layers, width 384, 6 heads | Trained jointly with the motion encoder |
| Latent compressor | 1 attention layer, attention width 3,072, 24 heads | 30,000 updates with the motion decoder, batch size 128 |
| Motion decoder | 4 Transformer layers, width 512, 8 heads | Trained with the compressor |
| Flow Transformer | 12 DiT blocks, width 512, 8 heads, feed-forward width 2,048 | Batch size 64 |
| Duration head | 1 hidden layer of width 512 with GELU | Trained jointly with the flow |

The motion encoder, predictor, compressor, and decoder use rotary temporal positions. We use AdamW with linear warm-up and cosine decay. The peak learning rate is $2\times10^{-4}$ for the motion stages and the initial generator training. The mBART encoder uses one tenth of the generator learning rate. We train the flow for 30,000 updates on Phoenix-2014T and 40,000 on How2Sign. On CSL-Daily, we train for 60,000 updates and initialize a further 40,000-update phase from the validation-selected checkpoint on the same training split. The continuation uses a peak learning rate of $1.5\times10^{-4}$ and retains the original CSL-Daily latent statistics. In each flow training phase, we select exponential moving average weights (decay 0.9999) by latent flow loss on a fixed 256-clip development subset.

### 3.1 Inference selection and evaluation

We select among single-sample settings on each full development split. A setting is eligible if both the upper-body and hand speed ratios are at least 0.5. Among eligible settings we minimize hand DTW-JPE, with body DTW-JPE as the tie-breaker. DTW-JPE accumulates mean Euclidean joint-position error along a dynamic time warping path. We measure 12 upper-body points relative to the pelvis and 21 points per hand relative to its wrist, and average the two hand scores. Speed ratios divide the generated mean joint speed by the reference mean speed after temporal Gaussian smoothing with a standard deviation of two frames. They track the amount of motion because a restrained trajectory can obtain a low position error under DTW. [3]

| Development split | Clips | Noise temperature | DTW-JPE body ↓ | DTW-JPE hand ↓ | Speed ratio, body | Speed ratio, hand |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phoenix-2014T | 519 | 1.0 | 5.18 | 6.90 | 0.71 | 0.77 |
| CSL-Daily | 1,077 | 0.7 | 5.88 | 7.78 | 0.67 | 0.65 |
| How2Sign | 1,722 | 1.0 | 7.10 | 10.21 | 0.82 | 1.00 |

All three rows use five Euler steps with guidance scale 1. For test evaluation, we fix the settings on development data and repeat training with the selected recipe. The completed runs yield body and hand DTW-JPE of 4.88 and 6.44 on Phoenix-2014T and 6.08 and 8.07 on CSL-Daily. These are single-seed results. DTW-JPE and the speed ratios characterize geometry and motion magnitude. The experiments reported here leave linguistic intelligibility unmeasured.

### 3.2 Phoenix-2014T ablations

We evaluated clean-latent prediction, motivated by SignNMFlow's use of target prediction on its motion manifold. In our Euclidean latent, this variant regresses the clean endpoint $z$ with a uniform squared-error loss. During sampling, at each step the predicted endpoint $\hat z$ defines the velocity $(\hat z-z_t)/(1-t)$. The update recomputes the endpoint from the new intermediate state at the next step. [15]

We also evaluated joint-position supervision with the velocity predictor. We recover the implied endpoint by adding the predicted velocity times the remaining integration time to $z_t$. For $t\geq0.3$, we decode this endpoint through the frozen decoder and convert the pose to 3D joints with differentiable SMPL-X forward kinematics. At 16 evenly spaced frame indices, smooth L1 penalties compare pelvis-relative body joints and wrist-relative hand joints against the original reference pose in centimetres. We weight the body and hand penalties by 0.25 and 1.0, respectively.

On the full development split, hand DTW-JPE was 6.97 for clean-latent prediction and 6.98 with joint-position supervision. Both runs use guidance scale 1.5 and temperature 0.7. The clean-latent run uses Euler updates on the implied velocity, and the joint-position run uses fourth-order Runge–Kutta. The velocity model trained without the joint-position loss scored 6.94 under the same Runge–Kutta settings and 6.90 under the selected Euler settings. The full-split comparison with clean-latent prediction changes the prediction target and sampling settings together.

In a separate check on 263 development clips, we drew eight motions at the predicted length for each sentence. The DTW medoid is the draw with the smallest average distance to the other seven. Starting from the medoid, DTW barycenter averaging repeatedly aligns the samples to the current estimate in joint space and averages the aligned pose parameters [16]. The medoid and barycenter yielded hand DTW-JPE of 7.08 and 6.79, respectively. The barycenter's body speed ratio was just below 0.5.

## References

1. Ben Saunders, Necati Cihan Camgoz, and Richard Bowden. ECCV, 2020. [Progressive Transformers for End-to-End Sign Language Production](https://doi.org/10.1007/978-3-030-58621-8_40).
2. Vasileios Baltatzis et al. CVPR, 2024. [Neural Sign Actors: A Diffusion Model for 3D Sign Language Production from Text](https://doi.org/10.1109/CVPR52733.2024.00194).
3. Ronglai Zuo, Rolandos Alexandros Potamias, Evangelos Ververas, Jiankang Deng, and Stefanos Zafeiriou. ICCV, 2025. [Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator](https://doi.org/10.1109/ICCV51701.2025.02210).
4. Xin Chen et al. CVPR, 2023. [Executing Your Commands via Motion Diffusion in Latent Space](https://doi.org/10.1109/CVPR52729.2023.01726).
5. Mahmoud Assran et al. CVPR, 2023. [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://doi.org/10.1109/CVPR52729.2023.01499).
6. Yifei Zhu et al. arXiv preprint, 2026. [MoRAE: Representation Autoencoders for Human Motion Generation](https://arxiv.org/html/2607.29180v2).
7. Georgios Pavlakos et al. CVPR, 2019. [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://doi.org/10.1109/CVPR.2019.01123).
8. Yaron Lipman et al. ICLR, 2023. [Flow Matching for Generative Modeling](https://openreview.net/pdf?id=PqvMRDCJT9t).
9. Yinhan Liu et al. TACL, 2020. [Multilingual Denoising Pre-training for Neural Machine Translation](https://doi.org/10.1162/tacl_a_00343).
10. William Peebles and Saining Xie. ICCV, 2023. [Scalable Diffusion Models with Transformers](https://doi.org/10.1109/ICCV51070.2023.00387).
11. Jonathan Ho and Tim Salimans. arXiv preprint, 2022. [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
12. Necati Cihan Camgoz, Simon Hadfield, Oscar Koller, Hermann Ney, and Richard Bowden. CVPR, 2018. [Neural Sign Language Translation](https://doi.org/10.1109/CVPR.2018.00812).
13. Hao Zhou, Wengang Zhou, Weizhen Qi, Junfu Pu, and Houqiang Li. CVPR, 2021. [Improving Sign Language Translation with Monolingual Data by Sign Back-Translation](https://doi.org/10.1109/CVPR46437.2021.00137).
14. Amanda Duarte et al. CVPR, 2021. [How2Sign: A Large-Scale Multimodal Dataset for Continuous American Sign Language](https://doi.org/10.1109/CVPR46437.2021.00276).
15. Jiayi He, Shengeng Tang, Sisi You, Yanbin Hao, Lechao Cheng, and Richang Hong. arXiv preprint, 2026. [Naturalness-guided Manifold Flow Matching for Sign Language Production](https://arxiv.org/abs/2609.33339).
16. François Petitjean, Alain Ketterlin, and Pierre Gançarski. Pattern Recognition, 2011. [A Global Averaging Method for Dynamic Time Warping, with Applications to Clustering](https://doi.org/10.1016/j.patcog.2010.09.013).
