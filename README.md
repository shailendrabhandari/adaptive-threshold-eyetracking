# adaptive-threshold-eyetracking
Adaptive fixation–saccade classification using K-ratio minimization

Code package for the paper:

> Orioma C, Krivan J, Mathema R, Lencastre P, Lind PG, Szorkovszky A, Bhandari S (2026).  
> *"Identification of fixations and saccades in eye-tracking data using adaptive threshold-based methods."*  
> arXiv: [2512.23926](https://arxiv.org/abs/2512.23926)

---

## What this package does

Implements three threshold-based fixation/saccade classifiers — **I-VT**, **I-AVT**, **I-DT** — with adaptive threshold selection via **K-ratio minimization**. Also includes:

- Gaussian noise robustness sweep 
- Markov adequacy diagnostics
Dataset: Random Pixel and Waldo (visual search) task — EyeLink 1000 Hz recordings.

---


## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/kratio-eyetracking.git
cd kratio-eyetracking
pip install -r requirements.txt
```

Place your Waldo `.txt` files in `data/extracted_waldo/` (see `data/README.md`).

Open the notebook:

```bash
jupyter notebook example_waldo.ipynb
```

---


## K-ratio definition

The K-ratio compares empirical fixation-to-saccade transition probability
to what would be expected if labels were independent:

$$K = \frac{n_{F \to S}}{n_S (1 - n_S)}$$

where $n_S$ = saccade fraction and $n_{F\to S}$ = empirical F→S transition rate.
Minimizing K over threshold values selects thresholds where fixations and saccades
form temporally coherent states.

---

## Citation

```bibtex
@article{orioma2026adaptive,
  title   = {Identification of fixations and saccades in eye-tracking data
             using adaptive threshold-based methods},
  author  = {Orioma, Charles and Krivan, Josef and Mathema, Rujeena and
             Lencastre, Pedro and Lind, Pedro G and Szorkovszky, Alexander and
             Bhandari, Shailendra},
  journal = {arXiv preprint arXiv:2512.23926},
  year    = {2026}
}
```

---
## Contributors

Josef Krivan initiated the codebase (project) and Charles Orioma extended it.  
Rujeena Mathema prepared and curated the dataset.  
Pedro Lencastre, Pedro G. Lind, Alexander Szorkovszky, and Shailendra Bhandari supervised the research.  
Shailendra Bhandari and Alexander Szorkovszky supervised the implementation, 
verified all formulas against the paper, and coordinated the public release.

For questions about the code package contact: shailendra.bhandari@oslomet.no 

## License

This project is licensed under the [MIT License](https://github.com/shailendrabhandari/adaptive-threshold-eyetracking/blob/main/LICENSE).

