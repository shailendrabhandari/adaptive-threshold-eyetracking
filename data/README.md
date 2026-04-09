# Data

Place your Waldo task EyeLink `.txt` files here in a subfolder, e.g.:

```
data/
└── extracted_waldo/
    ├── participant_01.txt
    ├── participant_02.txt
    └── ...
└──extracted_randomPixel/
    ├── participant_01.txt
    ├── participant_02.txt
    └── ...
```

Then set `DATA_DIR = "data/extracted_waldo"` in `example_waldo.ipynb`.

## File format

Each `.txt` file is a space-delimited EyeLink recording with 10 columns per sample at 1000 Hz:

| col_0 | col_1 | col_2 | col_3 | col_4 | col_5 | col_6 | col_7 | col_8 | col_9 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| timestamp | x_left | y_left | pupil_left | x_right | y_right | pupil_right | — | category_left | category_right |

- `category`: 0= blink, 1 = fixation, 2 = saccade (EyeLink default parser labels)

## Dataset reference

The full dataset is published in Scientific Data:

> Mathema R, Nav SM, Bhandari S, et al.  
> "Comprehensive dataset of features describing eye-gaze dynamics across multiple tasks."  
> *Scientific Data* (2026). https://doi.org/10.1038/s41597-026-06754-x
