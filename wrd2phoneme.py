

#!/usr/bin/env python3
"""
Convert the "wrd" field in a JSON file to canonical CMU phoneme sequences
and save them under a new key "canonical_cmu_phns".
Requires:
    pip install nltk
    python -c "import nltk; nltk.download('cmudict')"
Usage:
    python wrd2phoneme.py <input_json> [output_json]
If output_json is not provided, the input file will be overwritten.
"""
import json
import re
import sys
from nltk.corpus import cmudict

# Mapping from TIMIT-style phonemes to CMU format (lowercase)
timit2cmu = {
    "aa": "aa", "ae": "ae", "ah": "ah",
    "ao": "aa", "aw": "aw", "ay": "ay",
    "eh": "eh", "er": "er", "ey": "ey",
    "ih": "ih", "iy": "iy", "ow": "ow",
    "oy": "oy", "uh": "uh", "uw": "uw",
    "b": "b", "bcl": "b",
    "ch": "ch",
    "d": "d", "dcl": "d",
    "dh": "dh", "dx": "dx",
    "f": "f",
    "g": "g", "gcl": "g",
    "hh": "hh", "hv": "hh",
    "jh": "jh",
    "k": "k", "kcl": "k",
    "l": "l", "el": "l",
    "m": "m", "em": "m",
    "n": "n", "en": "n", "nx": "n",
    "ng": "ng",
    "p": "p", "pcl": "p",
    "r": "r",
    "s": "s",
    "sh": "sh",
    "t": "t", "tcl": "t",
    "th": "th",
    "v": "v",
    "w": "w",
    "y": "y",
    "z": "z",
    "zh": "zh",
    "pau": "sil", "h#": "sil", "epi": "sil",
    "cl": "sil", "q": "sil"
}

def word_to_cmu_phns(text, lexicon):
    """
    Convert a text string to a CMU phoneme sequence using nltk CMUDict
    and remap via timit2cmu.
    """
    phones = []
    words = re.findall(r"[A-Za-z']+", text)
    for w in words:
        key = w.lower()
        if key in lexicon:
            # take first pronunciation variant
            pron = lexicon[key][0]
            # strip stress digits and lowercase
            pron = [re.sub(r'\d', '', p).lower() for p in pron]
            # map TIMIT to CMU if needed
            pron = [timit2cmu.get(p, p) for p in pron]
            phones.extend(pron)
        else:
            # word not found in CMU dictionary
            print(f"Warning: '{w}' not in CMU lexicon, skipped")
    return " ".join(phones)

def main():
    if len(sys.argv) < 2:
        print("Usage: python wrd2phoneme.py <input_json> [output_json]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    # load CMU dictionary
    lexicon = cmudict.dict()

    # load JSON
    with open(input_path, 'r') as f:
        data = json.load(f)

    # update entries
    for utt, info in data.items():
        wrd = info.get("wrd", "")
        info["canonical_aligned"] = word_to_cmu_phns(wrd, lexicon)
        info["perceived_aligned"] = word_to_cmu_phns(wrd, lexicon)
        info["perceived_train_target"] = word_to_cmu_phns(wrd, lexicon)

    # save updated JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved updated JSON to {output_path}")

if __name__ == "__main__":
    main()