# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_boolean_decision_from_stats_each_round(stats_each_round):
    decisions = []
    for round_id, round_info in enumerate(stats_each_round):
        accepted_len = round_info["accepted_len"]

        if "~draft_proposal" in round_info and round_info["~draft_proposal"] is not None:
            spec_len = len(round_info["~draft_proposal"])
        else:
            spec_len = round_info.get("spec_len", 0)

        if accepted_len == spec_len:
            # print(f"Round {round_id} accepted all tokens!")
            decisions.extend([True] * (spec_len + 1))  # all accepted plus one bonus token
        else:
            # print(f"Round {round_id} accepted {accepted_len} tokens.")
            decisions.extend([True] * accepted_len)
            if spec_len > accepted_len:
                decisions.append(False)  # first rejected token
        
    # last_round = stats_each_round[-1]
    # last_round_prefix_len = last_round["prefix_len"]
    # last_round_accepted_tokens = last_round["accepted_len"]
    # if last_round["target_tokens"] == last_round["~draft_proposal"]:
    #     last_round_accepted_tokens += 1  # all accepted plus one bonus token
    # total_accepted_tokens = last_round_prefix_len + last_round_accepted_tokens
    # assert total_accepted_tokens <= len(decisions) <= total_accepted_tokens + 1, \
    #     f"Decisions length {len(decisions)}, total_accepted_tokens {total_accepted_tokens}!"
    return decisions


def visualize_acc_rate_over_time(stats_each_round, spec_len, acceptance_rate, figsize=(12, 3), output_dir=None, filename=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                    gridspec_kw={'height_ratios': [1, 2]}, 
                                    sharex=True)
    
    # bottom: acceptance rate at each round
    prefix_lens = [x.get("prefix_len", 0) + x["accepted_len"] for x in stats_each_round]
    acc_rates = []
    for x in stats_each_round:
        round_spec_len = x.get("spec_len")
        if round_spec_len is None and "~draft_proposal" in x and x["~draft_proposal"] is not None:
            round_spec_len = len(x["~draft_proposal"])
        if not round_spec_len:
            acc_rates.append(0.0)
        else:
            acc_rates.append(x["accepted_len"] / round_spec_len)
    decisions = get_boolean_decision_from_stats_each_round(stats_each_round)
    
    # print(f"prefix_lens: {prefix_lens}")
    # print(f"acc_rates: {acc_rates}")
    
    ax2.axhline(y=acceptance_rate, color="red", linestyle="dashed", zorder=1)
    ax2.plot(prefix_lens, acc_rates, marker='o', linewidth=2, label="Acceptance Rate")
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel("Token Index (prefix len when drafting)")
    ax2.set_ylabel("Acceptance Rate in that Round")
    ax2.grid(alpha=0.3)
    ax2.legend()
    y_ticks = np.linspace(0, 1, spec_len + 1)
    ax2.set_yticks(y_ticks)
    
    # top: raster plot
    arr = np.array(decisions, dtype=int)
    # series = pd.Series(arr)
    ax1.imshow(arr[np.newaxis, :], cmap='Greens', aspect='auto')
    ax1.set_yticks([])
    ax1.set_title("Accepted (green) vs Rejected (white)")
    
    plt.tight_layout()
    plt.show()
    if output_dir is not None and filename is not None:
        fig.savefig(os.path.join(output_dir, f"{filename}.png"))
        plt.close(fig)


def visualize_boolean_series(data, window=5, figsize=(10, 3), output_dir=None, problem_id=None):
    """
    Visualize a boolean time series with:
        - Top: raster (True = green, False = white)
        - Bottom: rolling average of True rate

    Args:
        data (list or np.ndarray): list of bools
        window (int): rolling window size for smoothing (default: 5)
        figsize (tuple): figure size (default: (10, 5))
    """
    # Convert to numeric array
    arr = np.array(data, dtype=int)
    series = pd.Series(arr)

    # Compute rolling mean
    rolling_rate = series.rolling(window, center=True).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                    gridspec_kw={'height_ratios': [1, 2]}, 
                                    sharex=True)

    # --- Top: Raster plot ---
    ax1.imshow(arr[np.newaxis, :], cmap='Greens', aspect='auto')
    ax1.set_yticks([])
    ax1.set_title("Accepted (green) vs Rejected (white)")
    
    # --- Bottom: Rolling average ---
    ax2.plot(series.index, rolling_rate, label=f"Rolling Acceptance Rate (window={window})", linewidth=2)
    ax2.scatter(series.index, arr, alpha=0.1, s=10, label="Raw (True=1, False=0)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Token Index")
    ax2.set_ylabel("True Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # fig.suptitle("Boolean Series Visualization", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    if output_dir is not None and problem_id is not None:
        fig.savefig(os.path.join(output_dir, f"{problem_id}.png"))
        plt.close(fig)

# %%
