# Homework 2 — A/B Testing with Multi-Armed Bandits  

A comparison of two bandit algorithms — **Epsilon-Greedy** and **Thompson Sampling**.  

**Note.** `report()` is a concrete shared method in the base class — it takes `data`, 
`selected_bandits`, and `algorithm_name` as parameters to serve both subclasses 
without duplication. All original abstract methods are preserved.

## Project Structure  

```  
Homework 2/  
├── bandit.py            # Main implementation  
├── BONUS.pdf            # Bonus section writeup  
├── exported_results/    # CSV output files from each algorithm run  
│   ├── E_Greedy_results.csv  
│   └── Thompson_results.csv  
├── README.md  
└── requirements.txt  
```  
  
## Setup  

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python bandit.py
```

This will run both experiments (20,000 trials each), print performance metrics to the console, save results to `exported_results/`, and display two plots.  

## Output

- **Console** — Cumulative reward, cumulative regret, and per-arm selection 
counts per algorithm  
- **Plot 1** — Average reward convergence over time (linear and log scale)  
- **Plot 2** — Cumulative rewards and cumulative regrets for both algorithms  
- **CSV files** — Per-trial log of `Bandit`, `Reward`, and `Algorithm`  

