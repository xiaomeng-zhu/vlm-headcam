from itertools import combinations
from pathlib import Path

import pandas as pd

DATA_DIR = Path("verb_data")
POS_IMAGE = "pos_url"
NEG_IMAGE = "neg_url"
GROUP = ["pos_subject", "pos_object", "pos_lemma"]

pairs = pd.read_csv(
    DATA_DIR / "svo_babyview_verb_pairs.csv",
    dtype={POS_IMAGE: str, NEG_IMAGE: str},
)
pairs = pairs.dropna(subset=[*GROUP, "neg_lemma", POS_IMAGE, NEG_IMAGE])

# Find images annotated with each lemma in either image role.
annotations = pd.concat([
    pairs[[POS_IMAGE, "pos_lemma"]].set_axis(["image", "lemma"], axis=1),
    pairs[[NEG_IMAGE, "neg_lemma"]].set_axis(["image", "lemma"], axis=1),
])
images_by_lemma = annotations.groupby("lemma")["image"].agg(set).to_dict()
annotations.groupby("lemma")["image"].agg(set).to_csv(DATA_DIR / "image_by_lemma.csv")


def can_build_trial(original_negatives, candidates):
    for paired_image, paired_lemma in original_negatives:
        remaining = [
            (image, lemma)
            for image, lemma in candidates
            if image != paired_image and lemma != paired_lemma
        ]
        for (image_a, lemma_a), (image_b, lemma_b) in combinations(remaining, 2):
            if image_a != image_b and lemma_a != lemma_b:
                return True
    return False


summary = []
for (subject, obj, target), group in pairs.groupby(GROUP):
    candidates = {
        (image, lemma)
        for image, lemma in group[[NEG_IMAGE, "neg_lemma"]].itertuples(
            index=False, name=None
        )
        if lemma != target and image not in images_by_lemma[target]
    }

    n_anchors = 0
    n_covered = 0
    for positive_image, anchor_rows in group.groupby(POS_IMAGE):
        n_anchors += 1
        original_negatives = set(
            anchor_rows[[NEG_IMAGE, "neg_lemma"]].itertuples(index=False, name=None)
        ) & candidates
        n_covered += can_build_trial(original_negatives, candidates)

    summary.append({
        "subject": subject,
        "object": obj,
        "target_lemma": target,
        "n_anchors": n_anchors,
        "n_covered_anchors": n_covered,
    })

coverage = pd.DataFrame(summary, columns=[
    "subject", "object", "target_lemma", "n_anchors", "n_covered_anchors"
])
covered_groups = coverage.loc[coverage["n_covered_anchors"].gt(0)]

print(f"Covered groups: {len(covered_groups):,} / {len(coverage):,}")
print(
    f"Covered anchors: {coverage['n_covered_anchors'].sum():,}"
    f" / {coverage['n_anchors'].sum():,}"
)
print(f"Target lemmas covered: {covered_groups['target_lemma'].nunique():,}")

coverage.to_csv(DATA_DIR / "svo_4afc_coverage_distinct_lemmas.csv", index=False)