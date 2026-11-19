import babyview_vlm as bv
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Load the model
model = bv.load("mcxfrank/babyview-vlm-ots-86m", seed=0)

# Load from their results the locations in tsne for all models
df = pd.read_csv("results/lexicon_tsne_B26_lad_base_s0.csv")
verb_list = df.loc[df["pos"] == "VERB", "word"].drop_duplicates().tolist()

words = [w for w in verb_list if w in model.vocab]
missing = [w for w in verb_list if w not in model.vocab]

ids = [model.vocab[w] for w in words] # ids for all words in the verb list
W = model.word.weight.detach().cpu().numpy()[ids].copy()

print(f"Found {len(words)} verbs; {len(missing)} missing from this model.") # found 212 verbs; none of them are missing from the model
print("Embedding shape:", W.shape)


# Obtain locations of embeddings in TSNE
W_unit = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8) # normalization

# Dimension reduction using tSNE
xy = TSNE(
    n_components=2,
    perplexity=min(30, len(words) - 1),
    metric="cosine",
    init="pca",
    random_state=0,
).fit_transform(W_unit)

points = pd.DataFrame({
    "word": words,
    "x": xy[:, 0],
    "y": xy[:, 1],
})


# Plotting
fig, ax = plt.subplots(figsize=(16, 12))
ax.scatter(points.x, points.y, s=18, alpha=0.7)

for row in points.itertuples():
    ax.annotate(
        row.word, (row.x, row.y),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=8,
    )

ax.set_title("Learned verb embeddings")
ax.set_axis_off()
fig.tight_layout()
fig.savefig("verb_figures/verb_tsne.png", dpi=250, bbox_inches="tight")
plt.show()

points.to_csv("verb_data/verb_tsne.csv", index=False)
np.savez_compressed(
    "verb_data/verb_embeddings.npz",
    words=np.asarray(words),
    W=W,
    W_normalized=W_unit,
)