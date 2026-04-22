# Dataset Licensing Notes

This file summarizes inferred dataset licensing constraints for this repository.

Legal note: this is a technical compliance aid, not legal advice.

## Scope

The project uses FER-style CSVs derived from individual dataset splits.
Code in this repository is MIT-licensed, but dataset rights are independent from code rights.

## Dataset Matrix (Inferred)

| Dataset | Source | Observed/Provided Terms | Commercial Use | Redistribution of Raw Data | Key Obligations | Confidence |
|---|---|---|---|---|---|---|
| FER2013 | https://www.kaggle.com/datasets/msambare/fer2013 | DbCL / Open Database Contents style terms | Likely Yes, subject to source terms | Conditional | Attribute source, keep database notice, check redistribution terms for derived CSVs and labels | Medium |
| Balanced Caer-S | https://www.kaggle.com/datasets/dollyprajapati182/balanced-caer-s-dataset-7575-grayscale | ODbL 1.0 | Yes, subject to ODbL obligations | Conditional | Attribution, keep ODbL notice, share-alike for public derivative databases, preserve notices | High |
| Emotic | https://arxiv.org/pdf/2003.13401 and https://s3.sunai.uoc.edu/emotic/index.html | Open source, but citation required | Unknown / likely research-first | Unknown / likely restricted by source terms | Cite the paper and dataset page; confirm whether redistribution of images/annotations is allowed | Medium |
| CK+ | Not used in current requested setup | Not applicable to this release if unused | Not applicable | Not applicable | Keep out of compliance scope if excluded from training/eval artifacts | High |

## Practical Interpretation

1. Derived CSVs inherit the strictest upstream constraints from the source dataset they are based on.
2. Mixing open-database licensed sources (for example ODbL/DBCL style obligations) with research-only or unknown-license sources can make public redistribution of derived row-level data legally risky or incompatible.
3. Publishing model code is usually simpler than publishing the underlying source datasets.
4. For this project, assume the safest interpretation for public release unless an official license page explicitly says otherwise.

## Release Policy Options

### Strict (Recommended)

1. Publish: code, configs, training scripts, metrics, and model cards.
2. Do not publish: raw third-party data, derived CSV rows, or row-level databases.
3. Add a notice: users must obtain each dataset from original providers under original terms.

### Medium

1. Publish checkpoints only if no source terms prohibit it.
2. Include dataset-attribution block and explicit non-commercial warning where applicable.
3. Keep derived training CSVs private unless all upstream licenses are confirmed compatible.

### Permissive (Only after verification)

1. Publish derivative databases only when every source permits it and obligations are met.
2. For ODbL-covered derivative databases, include ODbL notice and required access to derivative DB or modification method.
3. Document exact provenance and license compatibility mapping for each derived row source.

## Minimum Compliance Checklist

1. Record exact source URL and license text for each dataset version used.
2. Confirm whether commercial use is allowed per source.
3. Confirm whether redistribution is allowed for raw images, annotations, and derived CSVs.
4. Add required citations and attribution notices.
5. Add release README note: model/data rights differ; data must be re-downloaded by users.
6. Keep a frozen manifest of dataset versions and hashes used for each experiment.
7. Prefer citing both the paper and the hosting page when both exist.

## Suggested Repository Notice Snippet

"This repository code is MIT-licensed. Models are trained on third-party datasets
that are subject to their own licenses and terms. Users must obtain datasets from
original sources and comply with original dataset terms (including attribution,
non-commercial limits, and share-alike obligations where applicable)."