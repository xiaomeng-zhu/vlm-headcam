import argparse
from pathlib import Path

import babyview_vlm as bv
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent / "verb_data"
INPUT = DATA_DIR / "svo_verb_4afc_local.csv"
OUTPUT = DATA_DIR / "svo_verb_4afc_scores.csv"
PATH_COLUMNS = [f"image_{i}_path" for i in range(4)]


class LocalImages:
    """Load only the current batch of JPEGs requested by regions_hf()."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, batch_slice):
        images = []
        for path in self.paths[batch_slice]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mcxfrank/babyview-vlm-ots-86m")
    parser.add_argument("--encoder", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--include-cls", action="store_true")
    args = parser.parse_args()

    trials = pd.read_csv(INPUT)
    complete = trials["all_images_local"].astype(str).str.lower().eq("true")
    local_trials = trials.loc[complete].copy()
    model = bv.load(args.model, seed=args.seed, device=args.device)

    # Require the scorer to tokenize the target as exactly one vocabulary entry.
    in_vocab = local_trials["target_lemma"].map(
        lambda lemma: lemma in model.vocab
        and model.tokenize(lemma) == [model.vocab[lemma]]
    )
    eligible = local_trials.loc[in_vocab].copy()
    print(f"Trials with four local images: {len(local_trials):,} / {len(trials):,}")
    print(f"Trials excluded because the target cannot be scored: {(~in_vocab).sum():,}")
    if eligible.empty:
        raise SystemExit("No eligible trials to score.")

    # Deduplicate local paths and call regions_hf once so it loads DINOv3 once.
    relative_paths = list(dict.fromkeys(
        eligible[PATH_COLUMNS].to_numpy().ravel()
    ))
    image_index = {path: i for i, path in enumerate(relative_paths)}
    features = bv.regions_hf(
        LocalImages([DATA_DIR / path for path in relative_paths]),
        args.encoder,
        include_cls=args.include_cls,
        device=args.device,
        batch=args.batch_size,
    )

    results = []
    for row in tqdm(eligible.itertuples(index=False)):
        indices = [image_index[getattr(row, column)] for column in PATH_COLUMNS]
        scores = model.score([row.target_lemma], features[indices])[0].cpu().tolist()

        answer = row.correct_index
        prediction = max(range(4), key=scores.__getitem__)
        best_distractor = max(score for i, score in enumerate(scores) if i != answer)

        results.append({
            "trial_id": row.trial_id,
            "target_lemma": row.target_lemma,
            "correct_index": answer,
            "predicted_index": prediction,
            "correct": prediction == answer,
            "margin": scores[answer] - best_distractor,
            **{f"score_{i}": score for i, score in enumerate(scores)},
        })

    scored = pd.DataFrame(results)
    scored.to_csv(OUTPUT, index=False)
    print(f"Scored {len(scored):,} trials across {scored['target_lemma'].nunique()} verbs")
    print(f"4AFC accuracy: {scored['correct'].mean():.1%} (chance: 25%)")
    print(f"Mean cosine margin: {scored['margin'].mean():.4f}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()