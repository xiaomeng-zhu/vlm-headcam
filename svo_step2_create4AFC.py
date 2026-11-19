import json
import random
from collections import Counter
from http.client import HTTPException
from io import BytesIO
from itertools import combinations
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from tqdm import tqdm

import pandas as pd
from PIL import Image, UnidentifiedImageError

DATA_DIR = Path(__file__).resolve().parent / "verb_data"
INPUT = DATA_DIR / "svo_babyview_verb_pairs.csv"
OUTPUT = DATA_DIR / "svo_verb_4afc.csv"
CACHE = DATA_DIR / "svo_image_checks.jsonl"

POS_URL = "pos_url"
NEG_URL = "neg_url"
GROUP = ["pos_subject", "pos_object", "pos_lemma"]

SEED = 0
TIMEOUT = 8
MAX_BYTES = 20 * 1024 * 1024


def load_pairs():
    pairs = pd.read_csv(INPUT, dtype={POS_URL: str, NEG_URL: str})
    return pairs.dropna(subset=[*GROUP, "neg_lemma", POS_URL, NEG_URL])


def images_by_lemma(pairs):
    """Index images annotated with each lemma in either role."""
    positive = pairs[[POS_URL, "pos_lemma"]].set_axis(["url", "lemma"], axis=1)
    negative = pairs[[NEG_URL, "neg_lemma"]].set_axis(["url", "lemma"], axis=1)
    annotations = pd.concat([positive, negative], ignore_index=True)
    return annotations.groupby("lemma")["url"].agg(set).to_dict()


def negative_labels(rows):
    return set(rows[[NEG_URL, "neg_lemma"]].itertuples(index=False, name=None))


def distractor_options(original_negatives, candidates):
    """Yield three distinct images with three distinct lemmas."""
    for paired_url, paired_lemma in sorted(original_negatives):
        remaining = sorted(
            (url, lemma)
            for url, lemma in candidates
            if url != paired_url and lemma != paired_lemma
        )
        # list of all possible remaining two distractors
        for (url_a, lemma_a), (url_b, lemma_b) in combinations(remaining, 2):
            if url_a != url_b and lemma_a != lemma_b:
                yield (paired_url, paired_lemma), (url_a, lemma_a), (url_b, lemma_b)


def covered_anchors(pairs):
    """Find feasible anchors using annotations alone."""
    annotated_images = images_by_lemma(pairs)

    # start with the positive in each pair
    for (subject, obj, target), group in pairs.groupby(GROUP):
        candidates = {
            (url, lemma)
            for url, lemma in negative_labels(group)
            if lemma != target and url not in annotated_images[target]
        } # get all negative candidates for this positive group

        for positive_url, anchor_rows in group.groupby(POS_URL):
            original_negatives = negative_labels(anchor_rows) & candidates # the original negative paired with the positive
            options = list(distractor_options(original_negatives, candidates)) # a complete list of the distractor trios
            if options:
                yield subject, obj, target, positive_url, options


def check_image(url):
    """Download an image and verify it can be decoded."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return "invalid_url"

        request = Request(url, headers={"User-Agent": "SVO-Probes-image-check/1.0"})
        with urlopen(request, timeout=TIMEOUT) as response:
            data = response.read(MAX_BYTES + 1)
    except HTTPError as error:
        return "missing" if error.code in {404, 410} else "retry"
    except (URLError, OSError, HTTPException):
        return "retry"
    except ValueError:
        return "invalid_url"

    if len(data) > MAX_BYTES:
        return "too_large"

    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return "invalid_image"

    return "valid"


def load_cache():
    if not CACHE.exists():
        return {}

    with CACHE.open(encoding="utf-8") as file:
        records = (json.loads(line) for line in file if line.strip())
        return {record["url"]: record["status"] for record in records}


def image_is_valid(url, status_by_url, cache_file):
    if url not in status_by_url:
        status = check_image(url)
        status_by_url[url] = status

        # Temporary failures are retried on the next run.
        if status != "retry":
            cache_file.write(json.dumps({"url": url, "status": status}) + "\n")
            cache_file.flush()

    return status_by_url[url] == "valid"


def main():
    pairs = load_pairs()
    anchors = list(covered_anchors(pairs)) # everything to check
    print(f"Covered anchors before URL checks: {len(anchors):,}", flush=True)

    rng = random.Random(SEED)
    status_by_url = load_cache()
    outcomes = Counter()
    trials = []

    with CACHE.open("a", encoding="utf-8") as cache_file:
        for subject, obj, target, positive_url, options in tqdm(anchors, desc="anchoring..."):
            if not image_is_valid(positive_url, status_by_url, cache_file):
                outcomes["positive_unavailable"] += 1
                continue

            rng.shuffle(options)
            distractors = next( # get one option at a time if all urls are valid 
                (
                    option
                    for option in options
                    if all(
                        image_is_valid(url, status_by_url, cache_file)
                        for url, _ in option
                    )
                ),
                None,
            )
            if distractors is None:
                outcomes["no_available_distractor_combination"] += 1
                continue

            correct = (positive_url, target)
            choices = [correct, *distractors]
            rng.shuffle(choices)

            trial = {
                "trial_id": len(trials) + 1,
                "target_lemma": target,
                "subject": subject,
                "object": obj,
                "correct_index": choices.index(correct),
            }
            for index, (url, lemma) in enumerate(choices):
                trial[f"image_{index}"] = url
                trial[f"lemma_{index}"] = lemma
            trials.append(trial)

    columns = ["trial_id", "target_lemma", "subject", "object", "correct_index"]
    for index in range(4):
        columns.extend([f"image_{index}", f"lemma_{index}"])

    result = pd.DataFrame(trials, columns=columns)
    result.to_csv(OUTPUT, index=False)

    print(f"Anchors with unavailable positives: {outcomes['positive_unavailable']:,}")
    print(
        "Anchors without a usable distractor combination: "
        f"{outcomes['no_available_distractor_combination']:,}"
    )
    retry_count = sum(status == "retry" for status in status_by_url.values())
    print(f"URLs to retry on the next run: {retry_count:,}")
    print(f"Saved {len(result):,} trials across {result['target_lemma'].nunique()} verbs")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()