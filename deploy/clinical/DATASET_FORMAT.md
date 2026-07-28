# Clinical Dataset Format — what to ask hospitals for

The training pipeline (Stage 1 GANomaly + Stage 2 severity) consumes a simple
index: **`path, grade, source, split`** where `grade` is the DR severity 0–4.

Give hospitals **Format A** below — it is the easiest for them to export and
matches exactly how APTOS is laid out in this project. Format B is accepted
as an alternative.

---

## Format A (preferred): CSV + image folder

```
clinic_dataset/
├── labels.csv          # columns: id_code,diagnosis
└── images/
    ├── IMG_0001.jpg
    ├── IMG_0002.jpg
    └── ...
```

`labels.csv`:

```csv
id_code,diagnosis
IMG_0001,0
IMG_0002,3
IMG_0003,2
```

- `id_code` = image filename **without** extension (`.jpg`/`.jpeg`/`.png` all accepted)
- `diagnosis` = DR grade: `0` No DR, `1` Mild NPDR, `2` Moderate NPDR,
  `3` Severe NPDR, `4` Proliferative DR
- One row per image. Missing/duplicate ids are flagged by the converter.

(This is exactly the APTOS `train_1.csv` + `train_images/` layout.)

## Format B (alternative): folder per grade

```
clinic_dataset/
├── 0/   (healthy fundus images)
├── 1/   (mild NPDR)
├── 2/   (moderate NPDR)
├── 3/   (severe NPDR)
└── 4/   (proliferative DR)
```

(This is exactly the IDRiD layout.)

---

## Quality requirements to tell the hospital

1. **Labels come from an ophthalmologist** (or the hospital's grading
   protocol). Automated labels need a stated source.
2. Macula-centered fundus photos, any resolution ≥ 512px preferred; the
   pipeline crops/resizes itself.
3. Export images as JPG or PNG.
4. Note which images are **ungradable** (poor focus, artifacts) — either
   exclude them or flag them; do not silently include them in grade 0.
5. De-identification: filenames must not contain patient identifiers.

## Using the converter

```bash
# Format A:
python prepare_clinical_dataset.py --root /path/to/clinic_dataset --layout csv \
    --csv labels.csv --images images --out clinical_index.csv

# Format B:
python prepare_clinical_dataset.py --root /path/to/clinic_dataset --layout folders \
    --out clinical_index.csv
```

The script validates image readability, grade values, duplicates and class
balance, then writes `clinical_index.csv` with columns
`path,grade,source,split` (`source=clinical`, `split=all`).

## Then retrain

1. Upload `clinic_index.csv` + images to Kaggle as a dataset.
2. In each training notebook's data cell, add the clinical rows to `df`
   (same schema — one `pd.concat` line), or replace APTOS/ODIR entirely.
3. Run All → re-run the two export notebooks → replace `models/` on the Pi.
