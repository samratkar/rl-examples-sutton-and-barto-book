# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python implementations of examples and figures from Sutton & Barto's *Reinforcement Learning: An Introduction (2nd Edition)*. Each chapter directory contains self-contained scripts that reproduce specific figures/examples from the book.

## Running Scripts

Every script is self-contained and must be run from its own chapter directory (figures save to `../images/` via relative paths):

```bash
cd chapter02 && python ten_armed_testbed.py
```

Scripts use `matplotlib.use('Agg')` for non-interactive rendering and save PNG output to `images/`.

## Dependencies

```bash
pip install -r requirements.txt  # numpy, matplotlib, seaborn, tqdm
```

Local venv exists at `.venv/` (Python 3.14). The `.travis.yml` CI validates all scripts compile: `ls chapter*/*.py | xargs -n 1 -P 1 python -m py_compile`.

## Code Architecture

- **No shared library** -- each `.py` file is independent with no cross-file imports
- Scripts define environment dynamics, RL algorithms, and plotting in a single file
- Common pattern: `figure_X_Y()` or `example_X_Y()` functions that run simulations and call `plt.savefig()`, invoked from `if __name__ == '__main__'`
- Multi-run experiments use `tqdm` for progress bars and often parallelize with list comprehensions over runs
- Chapter 01 (`tic_tac_toe.py`) is unique: it's an interactive game, not a figure generator

## Conventions

- Figure naming follows the book: `figure_2_1.png`, `example_6_2.png`, `example_13_1.png`
- Keep the copyright declaration header at the top of each file
- RL hyperparameters (epsilon, alpha, gamma) are typically module-level constants
