import argparse
import ast
from pathlib import Path

import babyview_vlm as bv
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer

DATA_DIR = Path(__file__).resolve().parent / "verb_data"


def parse_triplet(value):
    if pd.isna(value):
        return None

    text = str(value).strip()
    if text.startswith("["):
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None

        if not isinstance(value, (list, tuple)) or len(value) != 1:
            return None
        text = str(value[0])

    parts = [part.strip().lower() for part in text.split(",")]
    return tuple(parts) if len(parts) == 3 and all(parts) else None


def get_verb_contrasts(dataset, lemmatizer):
    print(f"Total dataset: {len(dataset):,} rows")

    verb_negative = (
        dataset["verb_neg"].astype(str).str.strip().str.lower().eq("true")
    )
    verb_rows = dataset.loc[verb_negative].copy()
    print(f"Rows marked verb_neg=True: {len(verb_rows):,}")

    # triplets is a dictionary that maps pos/neg to the lists of triplets
    triplets = {
        side: verb_rows[f"{side}_triplet"].map(parse_triplet)
        for side in ("pos", "neg")
    }
    valid = triplets["pos"].notna() & triplets["neg"].notna()
    print(f"Unparseable triplet pairs excluded: {(~valid).sum():,}")

    pairs = verb_rows.loc[valid].copy()
    # breaks down the information in the triplet to different columns
    for side in ("pos", "neg"):
        columns = [f"{side}_{part}" for part in ("subject", "verb", "object")]
        pairs[columns] = pd.DataFrame(
            triplets[side].loc[valid].tolist(),
            index=pairs.index,
            columns=columns,
        )

    # make sure that the positive and negative examples have the same subject and the same object
    same_subject_object = (
        pairs["pos_subject"].eq(pairs["neg_subject"])
        & pairs["pos_object"].eq(pairs["neg_object"])
    )
    pairs = pairs.loc[same_subject_object].copy() # keep pairs with the same object and subjects
    print(f"Same subject and object: {len(pairs):,} rows")

    pairs = pairs.loc[pairs["pos_verb"].ne(pairs["neg_verb"])].copy() # keep pairs with different pos and neg verbs
    print(f"Also have different verbs: {len(pairs):,} rows")

    # create a lemma column for each verb
    for side in ("pos", "neg"):
        pairs[f"{side}_lemma"] = pairs[f"{side}_verb"].map(
            lambda word: lemmatizer.lemmatize(word, pos="v")
        )

    # only keep the pairs with different pos and neg lemmas
    pairs = pairs.loc[pairs["pos_lemma"].ne(pairs["neg_lemma"])].copy()
    print(f"Different verb lemmas: {len(pairs):,} rows")
    return pairs


def save_step0_subsets(pairs, lemmatizer):
    with np.load(DATA_DIR / "verb_embeddings.npz") as saved:
        words = saved["words"].astype(str).tolist()

    babyview_lemmas = {
        lemmatizer.lemmatize(word.strip().lower(), pos="v") for word in words
    } # the lemma of verb > 50 occurrences in babyview

    target = pairs.loc[pairs["pos_lemma"].isin(babyview_lemmas)]
    both = target.loc[target["neg_lemma"].isin(babyview_lemmas)] # the subset of svo probes where the pos and neg lemma are babyview lemmas

    print(
        f"Target lemma in Step 0 subset: {len(target):,} rows, "
        f"{target['pos_lemma'].nunique():,} target lemmas"
    )
    print(f"Both verb lemmas in Step 0 subset: {len(both):,} rows")

    target.to_csv(DATA_DIR / "svo_babyview_target_verbs.csv", index=False)
    both.to_csv(DATA_DIR / "svo_babyview_both_verbs.csv", index=False)

    return both


def save_shared_vocab_pairs(pairs, shared_vocab):
    lemmas_in_vocab = (
        pairs[["pos_lemma", "neg_lemma"]].isin(shared_vocab).all(axis=1)
    )
    pairs = pairs.loc[lemmas_in_vocab]
    print(f"Both verb lemmas in vocabulary: {len(pairs):,} rows")

    nouns_in_vocab = (
        pairs[["pos_subject", "pos_object"]].isin(shared_vocab).all(axis=1)
    ) # we can just check the pos_subject and pos_object because the neg_subject and neg_object are the same
    pairs = pairs.loc[nouns_in_vocab]
    print(f"Also have subject and object in vocabulary: {len(pairs):,} rows")

    pairs.to_csv(DATA_DIR / "svo_babyview_verb_pairs.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = pd.read_csv("svo_probes/svo_probes.csv") # load the svo probes dataset
    lemmatizer = WordNetLemmatizer()
    pairs = get_verb_contrasts(dataset, lemmatizer)
    both_in_step0 = save_step0_subsets(pairs, lemmatizer) # focus on the verbs > 50 occurences

    model = bv.load("mcxfrank/babyview-vlm-ots-86m", seed=args.seed)
    model_vocab = set(model.vocab)
    save_shared_vocab_pairs(both_in_step0, model_vocab)


if __name__ == "__main__":
    main()