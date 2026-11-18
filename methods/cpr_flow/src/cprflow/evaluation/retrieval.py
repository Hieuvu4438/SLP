import numpy as np


def retrieval_recall(motion, text, positive, ks=(1, 5, 10)):
    score = np.asarray(motion) @ np.asarray(text).T
    order = np.argsort(-score, axis=1, kind="stable")
    return {
        f"recall_at_{k}": float(np.take_along_axis(positive, order[:, :k], axis=1).any(1).mean())
        for k in ks
    }


def frechet_distance(real, generated):
    from scipy.linalg import sqrtm

    a, b = np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("FID requires equal fixed sample counts, at least two")
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root = sqrtm(ca @ cb)
    if np.iscomplexobj(root):
        if np.max(np.abs(root.imag)) > 1e-5:
            raise ValueError("Unstable covariance square root")
        root = root.real
    return float((ma - mb) @ (ma - mb) + np.trace(ca + cb - 2 * root))


def conditional_diversity(embeddings, prompt_ids, valid_semantic=None):
    """Mean pair distance per prompt; a semantic-subset claim requires real annotations."""
    if valid_semantic is None:
        return {
            "metric_status": "unavailable",
            "reason": "Missing validated semantic-subset annotations",
        }
    features = np.asarray(embeddings, dtype=np.float64)
    if len(features) != len(prompt_ids) or len(features) != len(valid_semantic):
        raise ValueError("Diversity feature/annotation lengths differ")
    groups = {prompt: [] for prompt in prompt_ids}
    for feature, prompt, valid in zip(features, prompt_ids, valid_semantic):
        if valid:
            groups[prompt].append(feature)
    scores = []
    for vectors in groups.values():
        if len(vectors) < 2:
            continue
        vectors = np.asarray(vectors)
        a, b = np.triu_indices(len(vectors), 1)
        scores.append(np.linalg.norm(vectors[a] - vectors[b], axis=-1).mean())
    return {
        "metric_status": "available" if scores else "unavailable",
        "mean_per_prompt": float(np.mean(scores)) if scores else None,
        "valid_prompts": len(scores),
        "requested_prompts": len(groups),
    }
