"""Highlight VerbNet classes on the saved Step 0 verb t-SNE map."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import verbnet
from nltk.stem import WordNetLemmatizer

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "verb_data"
FIGURE_DIR = BASE / "verb_figures"
MAX_CLASSES = 12


def load_points():
    points = pd.read_csv(
        DATA_DIR / "verb_tsne.csv",
        usecols=["word", "x", "y"],
        dtype={"word": str},
    )

    if points["word"].duplicated().any():
        raise ValueError("Coordinate file contains duplicate words.")

    return points


def annotate_memberships(points):
    lemmatizer = WordNetLemmatizer()

    # extract lemma of word
    points["lemma"] = points["word"].map(
        lambda word: lemmatizer.lemmatize(word, pos="v")
    )

    # map lemma to verb net classes
    classes = points["lemma"].map(
        lambda lemma: sorted(set(verbnet.classids(lemma=lemma)))
    )

    # Keep all words in the annotated table and all memberships in the counts.
    points["verbnet_classes"] = classes.str.join(";") # classes for the verb
    points["n_classes"] = classes.str.len() # how many classes is the verb in
    points["plot_class"] = points["verbnet_classes"].where(
        points["n_classes"].eq(1), "Multiple classes"
    )
    points.loc[points["n_classes"].eq(0), "plot_class"] = "Unmapped"

    members = points.assign(verbnet_class=classes).explode("verbnet_class") # create each row for a new class
    return members.dropna(subset=["verbnet_class"])


def plot_classes(points, members, top_classes):
    ncols = min(3, len(top_classes))
    nrows = (len(top_classes) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )

    for ax, class_id in zip(axes.flat, top_classes):
        group = members.loc[members["verbnet_class"].eq(class_id)]
        ax.scatter(points.x, points.y, s=12, color="lightgray", alpha=0.5)
        ax.scatter(group.x, group.y, s=35, color="tab:blue")
        for row in group.itertuples():
            ax.annotate(
                row.word, (row.x, row.y), xytext=(3, 3),
                textcoords="offset points", fontsize=8,
            )
        ax.set_title(f"{class_id} — {len(group)} words")
        ax.set_axis_off()

    for ax in axes.flat[len(top_classes):]:
        ax.set_visible(False)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "verbnet_largest_classes.png", dpi=250)
    fig.savefig(FIGURE_DIR / "verbnet_largest_classes.pdf")
    plt.close(fig)


def main():
    points = load_points()
    members = annotate_memberships(points)

    # counting the number of items in each class to decide what to plot
    counts = (
        members.groupby("verbnet_class")["word"]
        .nunique()
        .sort_values(ascending=False, kind="stable")
        .rename("n_words")
    )
    points.to_csv(DATA_DIR / "verb_tsne_verbnet.csv", index=False)
    counts.to_csv(DATA_DIR / "verbnet_class_counts.csv")

    top_classes = counts.head(MAX_CLASSES)
    if top_classes.empty:
        raise ValueError("No VerbNet memberships found.")
    print(top_classes)
    plot_classes(points, members, top_classes.index.tolist())


if __name__ == "__main__":
    main()
