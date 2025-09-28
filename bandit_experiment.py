import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from bandit_base import ContextProcessor
from bandit_algorithms import (
    LinUCB,
    SlidingWindowLinUCB,
    LinUCBDecay,
    LinTS,
    EpsilonGreedy,
    SlidingDoublyRobustSoftmax,
    RidgeSoftmax,
)


class BanditExperiment:
    """Class to run and evaluate bandit algorithms on the synthetic data."""

    def __init__(self, data, algorithms=None):
        """
        Initialize the experiment.

        Args:
            data: List of dictionaries with 'context' and 'rewards'
            algorithms: List of initialized contextual bandit algorithms
        """
        self.data = data
        self.algorithms = algorithms or []

        # Extract the number of arms
        self.n_arms = len(data[0]["rewards"])

        # Preprocess contexts
        self.context_processor = ContextProcessor()
        self.contexts = [d["context"] for d in data]
        self.context_dim = self.context_processor.fit(self.contexts)

        # Set context processor for each algorithm
        for algo in self.algorithms:
            algo.context_processor = self.context_processor

    def add_algorithm(self, algorithm_class, **kwargs):
        """
        Add an algorithm to the experiment.

        Args:
            algorithm_class: Class of the algorithm to add
            **kwargs: Arguments to pass to the algorithm constructor
        """
        algo = algorithm_class(self.n_arms, self.context_dim, **kwargs)
        algo.context_processor = self.context_processor
        self.algorithms.append(algo)

    def run(self):
        """Run the experiment on all algorithms."""
        results = {}

        # Track optimal and worst cumulative reward
        optimal_cumulative_rewards = []
        worst_cumulative_rewards = []
        optimal_cum_reward = 0
        worst_cum_reward = 0

        print(f"Starting experiment with {len(self.algorithms)} algorithms")
        for i, algo in enumerate(tqdm(self.algorithms, desc="Running algorithms")):
            print(f"\nProcessing algorithm {i}: {algo.name}")

            # Store current parameters and DataFrames before reset
            if isinstance(algo, SlidingWindowLinUCB):  # Check SlidingWindowLinUCB first
                alpha = algo.alpha
                window_size = algo.window_size
                self.algorithms[i] = SlidingWindowLinUCB(
                    algo.n_arms, algo.context_dim, alpha=alpha, window_size=window_size
                )
                print(f"Reset SlidingWindowLinUCB with alpha={alpha}, window_size={window_size}")
            elif isinstance(algo, LinUCB):
                alpha = algo.alpha
                self.algorithms[i] = LinUCB(algo.n_arms, algo.context_dim, alpha=alpha)
                print(f"Reset LinUCB with alpha={alpha}")
            elif isinstance(algo, LinUCBDecay):
                alpha = algo.alpha
                decay = algo.decay
                self.algorithms[i] = LinUCBDecay(
                    algo.n_arms, algo.context_dim, alpha=alpha, decay=decay
                )
                print(f"Reset LinUCBDecay with alpha={alpha}, decay={decay}")
            elif isinstance(algo, LinTS):
                v = algo.v
                self.algorithms[i] = LinTS(algo.n_arms, algo.context_dim, v=v)
                print(f"Reset LinTS with v={v}")
            elif isinstance(algo, EpsilonGreedy):
                epsilon = algo.epsilon
                self.algorithms[i] = EpsilonGreedy(algo.n_arms, algo.context_dim, epsilon=epsilon)
                print(f"Reset EpsilonGreedy with epsilon={epsilon}")
            elif isinstance(algo, SlidingDoublyRobustSoftmax):
                tau = algo.tau
                window_size = algo.window_size
                lambda_reg = algo.lambda_reg
                # Store the DataFrames
                selection_log = algo.selection_log_df.copy()
                update_log = algo.update_log_df.copy()
                # Create new instance and update reference
                self.algorithms[i] = SlidingDoublyRobustSoftmax(
                    algo.n_arms,
                    algo.context_dim,
                    tau=tau,
                    window_size=window_size,
                    lambda_reg=lambda_reg,
                )
                # Restore the DataFrames
                self.algorithms[i].selection_log_df = selection_log
                self.algorithms[i].update_log_df = update_log
                print(f"Reset SlidingDoublyRobustSoftmax with tau={tau}, window_size={window_size}")
            elif isinstance(algo, RidgeSoftmax):
                tau = algo.tau
                lambda_reg = algo.lambda_reg
                selection_log = algo.selection_log_df.copy()
                update_log = algo.update_log_df.copy()
                self.algorithms[i] = RidgeSoftmax(
                    algo.n_arms, algo.context_dim, tau=tau, lambda_reg=lambda_reg
                )
                print(f"Reset RidgeSoftmax with tau {tau}")

            # Set context processor
            self.algorithms[i].context_processor = self.context_processor

            # Track cumulative rewards and regrets
            cumulative_rewards = []
            cumulative_regrets = []

            # Initialize arm_counts with the maximum possible size (n_arms + 1 for the new arm)
            arm_counts = np.zeros(self.n_arms + 1)

            # Flag to track if we've added the new arm
            new_arm_added = False

            # Run algorithm on data
            for j, entry in enumerate(
                tqdm(self.data, desc=f"Running {self.algorithms[i].name}", leave=False)
            ):
                context = entry["context"]
                rewards = entry["rewards"]

                # Check if we need to add the new arm
                if not new_arm_added and context["date"] == "2023-06-30":
                    self.algorithms[i].add_arm()
                    new_arm_added = True
                    print(f"Added new arm to {self.algorithms[i].name} at step {j}")

                # Select arm
                arm = self.algorithms[i].select_arm(context)

                # Get reward for selected arm
                reward = rewards[str(arm)]

                # Update algorithm
                self.algorithms[i].update(context, arm, reward)

                # Track metrics
                cumulative_rewards.append(self.algorithms[i].cumulative_reward)

                # Calculate regret (difference from optimal reward)
                optimal_arm = max(rewards.items(), key=lambda x: x[1])[0]
                worst_arm = min(rewards.items(), key=lambda x: x[1])[0]
                optimal_reward = rewards[optimal_arm]
                worst_reward = rewards[worst_arm]

                # Track optimal and worst cumulative reward (only need to do this once)
                if len(optimal_cumulative_rewards) <= j:
                    optimal_cum_reward += optimal_reward
                    worst_cum_reward += worst_reward
                    optimal_cumulative_rewards.append(optimal_cum_reward)
                    worst_cumulative_rewards.append(worst_cum_reward)

                regret = optimal_reward - reward
                self.algorithms[i].cumulative_regret += regret
                cumulative_regrets.append(self.algorithms[i].cumulative_regret)

                # Track arm selection counts
                arm_counts[arm] += 1

            # Store results
            results[self.algorithms[i].name] = {
                "cumulative_rewards": cumulative_rewards,
                "cumulative_regrets": cumulative_regrets,
                "arm_counts": arm_counts,
                "arm_frequencies": arm_counts / len(self.data),
                "selected_arms": self.algorithms[i].selected_arms,
                "obtained_rewards": self.algorithms[i].obtained_rewards,
            }
            print(f"Completed running {self.algorithms[i].name}")
            print(f"Final cumulative reward: {cumulative_rewards[-1]:.2f}")
            print(f"Final cumulative regret: {cumulative_regrets[-1]:.2f}")

        self.results = results
        self.optimal_cumulative_rewards = optimal_cumulative_rewards
        self.worst_cumulative_rewards = worst_cumulative_rewards

        # Print final results summary
        print("\nFinal Results Summary:")
        for algo_name, result in results.items():
            print(f"\n{algo_name}:")
            print(f"Final cumulative reward: {result['cumulative_rewards'][-1]:.2f}")
            print(f"Final cumulative regret: {result['cumulative_regrets'][-1]:.2f}")
            print(f"Mean reward: {np.mean(result['obtained_rewards']):.2f}")

        return results

    def plot_cumulative_rewards(self):
        """Plot cumulative rewards for all algorithms."""
        plt.figure(figsize=(12, 6))

        # Plot optimal and worst cumulative reward
        plt.plot(self.optimal_cumulative_rewards, "k--", label="Optimal", alpha=0.7)
        plt.plot(self.worst_cumulative_rewards, "r--", label="Worst", alpha=0.7)

        for algo_name, result in self.results.items():
            plt.plot(result["cumulative_rewards"], label=algo_name)

        plt.xlabel("Time step")
        plt.ylabel("Cumulative reward")
        plt.title("Cumulative Reward Comparison")
        plt.legend()

        # Add more grid lines and adjust y-axis
        plt.grid(True, alpha=0.3, linestyle="--")

        # Calculate y-axis limits with some padding
        y_min = min(min(result["cumulative_rewards"]) for result in self.results.values())
        y_max = max(max(result["cumulative_rewards"]) for result in self.results.values())
        y_range = y_max - y_min
        padding = y_range * 0.1  # 10% padding

        plt.ylim(y_min - padding, y_max + padding)

        # Add more y-axis ticks
        plt.yticks(np.arange(y_min - padding, y_max + padding, y_range / 10))

        plt.tight_layout()

        return plt.gcf()

    def plot_cumulative_regrets(self):
        """Plot cumulative regrets for all algorithms."""
        plt.figure(figsize=(12, 6))

        for algo_name, result in self.results.items():
            plt.plot(result["cumulative_regrets"], label=algo_name)

        plt.xlabel("Time step")
        plt.ylabel("Cumulative regret")
        plt.title("Cumulative Regret Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt.gcf()

    def plot_arm_selection_frequencies(self):
        """Plot arm selection frequencies for all algorithms."""
        n_algos = len(self.algorithms)
        fig, axes = plt.subplots(1, n_algos, figsize=(15, 5), sharey=True)

        if n_algos == 1:
            axes = [axes]

        for i, (algo_name, result) in enumerate(self.results.items()):
            n_frequencies = len(result["arm_frequencies"])
            axes[i].bar(range(n_frequencies), result["arm_frequencies"])
            axes[i].set_title(algo_name)
            axes[i].set_xlabel("Arm")
            if i == 0:
                axes[i].set_ylabel("Selection frequency")

        # Add suptitle *before* tight_layout and reserve space with rect
        fig.suptitle("Arm Selection Frequencies per Algorithm", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave space at top for suptitle
        return fig

    def plot_arm_selections_over_time(self):
        """Plot arm selections over time for all algorithms."""
        n_algos = len(self.algorithms)
        fig, axes = plt.subplots(n_algos, 1, figsize=(15, 3 * n_algos), sharex=True)

        if n_algos == 1:
            axes = [axes]

        for i, (algo_name, result) in enumerate(self.results.items()):
            # Scatter plot of arm selections
            axes[i].scatter(
                range(len(result["selected_arms"])), result["selected_arms"], s=1, alpha=0.5
            )

            # Add regime shift markers if available
            regime_shifts = self._find_regime_shifts()
            for shift in regime_shifts:
                axes[i].axvline(x=shift, color="r", linestyle="--", alpha=0.7)

            axes[i].set_title(f"{algo_name} - Arm Selections Over Time")
            axes[i].set_ylabel("Selected Arm")
            axes[i].set_ylim(-0.5, self.n_arms - 0.5)
            axes[i].set_yticks(range(self.n_arms))

        axes[-1].set_xlabel("Time step")
        plt.tight_layout()

        return fig

    def _find_regime_shifts(self):
        """
        Find potential regime shifts based on date changes.
        Returns indices of regime shifts.
        """
        dates = [entry["context"]["date"] for entry in self.data]
        shifts = []

        # Look for specific dates - modify based on your dataset
        # This assumes regime shifts occur at days 90 and 240 as mentioned in your dissertation
        all_dates = sorted(list(set(dates)))  # Get unique dates

        # Find days 90 and 240
        if len(all_dates) >= 90:
            day_90 = all_dates[89]  # 0-indexed
            day_90_idx = dates.index(day_90)
            shifts.append(day_90_idx)

        if len(all_dates) >= 240:
            day_240 = all_dates[239]  # 0-indexed
            day_240_idx = dates.index(day_240)
            shifts.append(day_240_idx)

        # Also look for cold start day 180
        if len(all_dates) >= 180:
            day_180 = all_dates[179]  # 0-indexed
            day_180_idx = dates.index(day_180)
            shifts.append(day_180_idx)

        return shifts

    def print_summary_statistics(self):
        """Print summary statistics for all algorithms."""
        # Create a comparison dataframe
        stats = []

        for algo_name, result in self.results.items():
            final_reward = result["cumulative_rewards"][-1]
            final_regret = result["cumulative_regrets"][-1]
            mean_reward = np.mean(result["obtained_rewards"])

            stats.append(
                {
                    "Algorithm": algo_name,
                    "Final Cumulative Reward": final_reward,
                    "Final Cumulative Regret": final_regret,
                    "Mean Reward": mean_reward,
                    "Regret per Step": final_regret / len(self.data),
                }
            )

        stats_df = pd.DataFrame(stats)
        return stats_df

    def calculate_adaptation_metrics(self):
        """
        Calculate metrics related to adaptation to non-stationarity.

        Returns:
            DataFrame with adaptation metrics
        """
        # Find regime shifts
        regime_shifts = self._find_regime_shifts()
        print(f"Found regime shifts at indices: {regime_shifts}")

        if not regime_shifts:
            print("No regime shifts detected. Cannot calculate adaptation metrics.")
            return None

        # Calculate adaptation speed for each algorithm
        adaptation_metrics = []

        for algo_name, result in self.results.items():
            print(f"\nProcessing algorithm: {algo_name}")
            rewards = result["obtained_rewards"]
            print(f"Number of rewards: {len(rewards)}")

            for i, shift in enumerate(regime_shifts):
                print(f"Processing shift {i+1} at index {shift}")
                # Skip if shift is too close to the end
                if shift + 100 >= len(rewards):
                    print(f"Skipping shift {i+1} - too close to end of data")
                    continue

                # Calculate metrics around the shift
                pre_shift = rewards[max(0, shift - 100) : shift]
                post_shift = rewards[shift : shift + 100]

                # Calculate adaptation metrics
                pre_mean = np.mean(pre_shift) if pre_shift else 0
                post_mean = np.mean(post_shift)

                recovery_time = None
                window_size = 10
                for j in range(window_size, len(post_shift)):
                    window_mean = np.mean(post_shift[j - window_size : j])
                    if window_mean > pre_mean:
                        recovery_time = j
                        break

                print(
                    f"Pre-shift mean: {pre_mean:.4f}, Post-shift mean: {post_mean:.4f}, Recovery time: {recovery_time}"
                )

                adaptation_metrics.append(
                    {
                        "Algorithm": algo_name,
                        "Regime Shift": i + 1,
                        "Shift Time Step": shift,
                        "Pre-Shift Mean Reward": pre_mean,
                        "Post-Shift Mean Reward": post_mean,
                        "Recovery Time (steps)": recovery_time,
                    }
                )

        if not adaptation_metrics:
            print("No adaptation metrics were collected.")
            return None

        adaptation_df = pd.DataFrame(adaptation_metrics)
        print(f"\nCreated DataFrame with {len(adaptation_df)} rows")
        return adaptation_df

    def analyze_cold_start(self):
        """
        Analyze how algorithms handle the cold-start problem.
        """
        print("Analyzing cold-start behavior...")

        # Find when a new arm/strategy is introduced
        dates = [entry["context"]["date"] for entry in self.data]
        all_dates = sorted(list(set(dates)))  # Get unique dates

        if len(all_dates) < 180:
            print("Dataset too small to find day 180 for cold start analysis.")
            return pd.DataFrame([])

        day_180 = all_dates[179]  # 0-indexed

        try:
            cold_start_idx = dates.index(day_180)
            print(f"Cold start index (day 180): {cold_start_idx}")
        except ValueError:
            cold_start_idx = len(dates) // 2
            print(f"Exact cold start day not found. Using approximate index: {cold_start_idx}")

        cold_start_metrics = []

        for algo_name, result in self.results.items():
            selections = result["selected_arms"]

            new_arm = min(4, self.n_arms - 1)  # Use last arm if fewer than 5

            try:
                first_selection = selections.index(new_arm, cold_start_idx)
                time_to_first_selection = first_selection - cold_start_idx
            except ValueError:
                time_to_first_selection = float("inf")

            if cold_start_idx + 100 < len(selections):
                post_intro_selections = selections[cold_start_idx : cold_start_idx + 100]
                new_arm_frequency = post_intro_selections.count(new_arm) / 100
            else:
                post_intro_selections = selections[cold_start_idx:]
                if post_intro_selections:
                    new_arm_frequency = post_intro_selections.count(new_arm) / len(
                        post_intro_selections
                    )
                else:
                    new_arm_frequency = 0.0

            cold_start_metrics.append(
                {
                    "Algorithm": algo_name,
                    "Time to First Selection": time_to_first_selection,
                    "Selection Frequency (first 100 steps)": new_arm_frequency,
                }
            )

        cold_start_df = pd.DataFrame(cold_start_metrics)

        if cold_start_df.empty:
            print("No cold start data could be collected.")
        else:
            print(f"Cold start metrics collected for {len(cold_start_df)} algorithms.")

        return cold_start_df

    def statistical_tests(self, alpha=0.05):
        """
        Perform statistical tests to compare algorithm performance.

        Args:
            alpha (float): Significance level for statistical tests (default: 0.05)

        Returns:
            tuple: (test_results, tukey_summary)
                - test_results: Dictionary containing ANOVA results
                - tukey_summary: DataFrame with pairwise comparisons if ANOVA is significant
        """
        # Extract rewards for each algorithm
        rewards_by_algo = {
            algo_name: result["obtained_rewards"] for algo_name, result in self.results.items()
        }

        if not rewards_by_algo:
            raise ValueError("No algorithm results available for statistical testing")

        # Ensure all arrays have the same length (truncate to shortest)
        min_length = min(len(rewards) for rewards in rewards_by_algo.values())
        if min_length < 2:
            raise ValueError("Insufficient data points for statistical testing")

        rewards_by_algo = {algo: rewards[:min_length] for algo, rewards in rewards_by_algo.items()}

        # Check for missing values
        for algo, rewards in rewards_by_algo.items():
            if any(np.isnan(rewards)):
                print(f"Warning: Found missing values in rewards for {algo}")
                rewards_by_algo[algo] = [r for r in rewards if not np.isnan(r)]

        # Perform ANOVA to test for significant differences
        algorithm_names = list(rewards_by_algo.keys())
        all_rewards = [rewards_by_algo[algo] for algo in algorithm_names]

        # One-way ANOVA
        f_val, p_val = stats.f_oneway(*all_rewards)

        # Convert numpy types to Python native types for the test results
        test_results = {
            "ANOVA F-value": float(f_val),
            "ANOVA p-value": float(p_val),
            "Significant Difference": bool(p_val < alpha),
            "Number of Algorithms": len(algorithm_names),
            "Data Points per Algorithm": min_length,
        }

        # If significant, perform Tukey's HSD test
        if p_val < alpha:
            # Prepare data for tukey test
            data = []
            labels = []

            for algo, rewards in rewards_by_algo.items():
                data.extend(rewards)
                labels.extend([algo] * len(rewards))

            # Perform Tukey's test
            tukey_result = pairwise_tukeyhsd(data, labels, alpha=alpha)

            # Convert the tukey result to a DataFrame
            tukey_summary = pd.DataFrame(
                data=tukey_result._results_table.data[1:],  # Skip the header row
                columns=tukey_result._results_table.data[0],  # Use header row as column names
            )

            # Convert numpy types to Python native types in the DataFrame
            for col in tukey_summary.columns:
                if tukey_summary[col].dtype in [np.float64, np.float32]:
                    tukey_summary[col] = tukey_summary[col].astype(float)
                elif tukey_summary[col].dtype in [np.int64, np.int32]:
                    tukey_summary[col] = tukey_summary[col].astype(int)
                elif tukey_summary[col].dtype == np.bool_:
                    tukey_summary[col] = tukey_summary[col].astype(bool)

            # Add mean values for easier interpretation
            algo_means = {
                algo: float(np.mean(rewards)) for algo, rewards in rewards_by_algo.items()
            }
            tukey_summary["Mean 1"] = [algo_means[group1] for group1 in tukey_summary["group1"]]
            tukey_summary["Mean 2"] = [algo_means[group2] for group2 in tukey_summary["group2"]]

            # Add interpretation column
            tukey_summary["Interpretation"] = tukey_summary.apply(
                lambda row: f"{row['group1']} is {'significantly' if row['reject'] else 'not significantly'} different from {row['group2']}",
                axis=1,
            )

            return test_results, tukey_summary

        return test_results, None

    def bootstrap_confidence_intervals(
        self, metric="cumulative_regret", n_bootstrap=1000, alpha=0.05
    ):
        """
        Calculate bootstrap confidence intervals for performance metrics.

        Args:
            metric: 'cumulative_regret' or 'cumulative_reward'
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level

        Returns:
            DataFrame with bootstrap confidence intervals
        """
        bootstrap_results = []

        for algo_name, result in self.results.items():
            if metric == "cumulative_regret":
                final_value = result["cumulative_regrets"][-1]
                values = result["cumulative_regrets"]
            else:  # cumulative_reward
                final_value = result["cumulative_rewards"][-1]
                values = result["cumulative_rewards"]

            # Create bootstrap samples
            bootstrap_samples = []
            for _ in range(n_bootstrap):
                # Resample with replacement
                indices = np.random.choice(len(values), size=len(values), replace=True)
                resampled = [values[i] for i in indices]
                bootstrap_samples.append(resampled[-1])  # Final value of resampled trajectory

            # Calculate confidence intervals
            lower_ci = np.percentile(bootstrap_samples, alpha / 2 * 100)
            upper_ci = np.percentile(bootstrap_samples, (1 - alpha / 2) * 100)

            bootstrap_results.append(
                {
                    "Algorithm": algo_name,
                    f'Final {metric.replace("_", " ").title()}': final_value,
                    "Lower CI": lower_ci,
                    "Upper CI": upper_ci,
                    "CI Width": upper_ci - lower_ci,
                }
            )

        bootstrap_df = pd.DataFrame(bootstrap_results)
        return bootstrap_df

    def plot_mean_std_multiple(self, results_dict, trades_per_day=50):
        """
        Plot mean ± std for multiple algorithms from Monte Carlo results.

        Args:
            results_dict: Dict of {algo_name: np.ndarray of shape (n_runs, n_steps)}
        """
        plt.figure(figsize=(12, 6))
        for algo_name, reward_matrix in results_dict.items():
            mean = np.mean(reward_matrix, axis=0)
            std = np.std(reward_matrix, axis=0)
            days = np.arange(mean.shape[0]) / trades_per_day
            plt.plot(days, mean, label=algo_name)
            plt.fill_between(days, mean - std, mean + std, alpha=0.2)

        plt.title("Monte Carlo Simulation: Mean ± Std of Cumulative Reward")
        plt.xlabel("Days")
        plt.ylabel("Cumulative Reward")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def generate_cold_start_auc_and_strategy_distribution(
        self, evaluation_period_days=30, trades_per_day=50
    ):
        """
        Generate metrics and figures for strategy selection distribution and cold-start performance.

        Strategy Selection Distribution (Equation 12):
            P^a(k) = (Number of times algorithm a selected strategy k) / (Total number of decisions)

        Cold-Start Performance (Equation 13):
            ColdStartAUC^a = (1 / T_new) * ∑[t=t_intro}^{t=t_intro+T_new] 1_{a_t = a_new} · r_t
            where t_intro is the time step when the new strategy is introduced and T_new is the evaluation period.

        The method computes for each algorithm:
        - A strategy distribution table (arm index and selection frequency).
        - A cold-start metrics table including:
            * Time to first new arm selection within the evaluation window
            * New arm selection count
            * Exploration rate (fraction of selections of the new arm)
            * Cold-Start AUC (average reward from selecting the new arm during evaluation)
        - A bar plot showing strategy selection distributions per algorithm.
        - A line plot of the cumulative new arm reward over the evaluation window.

        Parameters:
            evaluation_period_days (int): Number of days in the evaluation period (default 30).
            trades_per_day (int): Number of decisions (trades) per day (default 50).

        Returns:
            dict: {
                'strategy_distribution_table': DataFrame,
                'strategy_distribution_plot': matplotlib.figure.Figure,
                'cold_start_metrics_table': DataFrame,
                'cold_start_auc_plot': matplotlib.figure.Figure
            }
        """

        # Determine evaluation period in steps
        evaluation_period_steps = evaluation_period_days * trades_per_day

        # Determine cold start index using day 180 (i.e., the 180th unique date)
        dates = [entry["context"]["date"] for entry in self.data]
        all_dates = sorted(list(set(dates)))
        if len(all_dates) < 180:
            print("Not enough data to determine cold start day.")
            return None
        cold_start_date = all_dates[179]  # 0-indexed: day 180
        cold_start_idx = dates.index(cold_start_date)
        print(
            f"Cold start analysis starting at index {cold_start_idx} (date {cold_start_date}) with evaluation period of {evaluation_period_steps} steps."
        )

        # Initialize containers for metrics
        cold_start_metrics_list = []
        strategy_distribution_list = []
        cumulative_new_arm_rewards = {}

        # Iterate through each algorithm's results
        for algo_name, result in self.results.items():
            # Strategy distribution: use arm_frequencies already computed
            freqs = result["arm_frequencies"]
            for arm_index, freq in enumerate(freqs):
                strategy_distribution_list.append(
                    {"Algorithm": algo_name, "Arm": arm_index, "Selection Frequency": freq}
                )

            # Determine new arm index (assumed to be the last arm in the frequency array)
            new_arm_index = len(freqs) - 1

            # Extract evaluation window for cold start analysis
            window_selections = result["selected_arms"][
                cold_start_idx : cold_start_idx + evaluation_period_steps
            ]
            window_rewards = result["obtained_rewards"][
                cold_start_idx : cold_start_idx + evaluation_period_steps
            ]

            # Calculate exploration rate for the new arm
            new_arm_selections = [1 if arm == new_arm_index else 0 for arm in window_selections]
            exploration_rate = np.sum(new_arm_selections) / evaluation_period_steps

            # Calculate Cold-Start AUC: average reward from selecting the new arm over the evaluation period
            new_arm_rewards = [
                reward if arm == new_arm_index else 0
                for arm, reward in zip(window_selections, window_rewards)
            ]
            cold_start_auc = np.sum(new_arm_rewards) / evaluation_period_steps

            # Determine time to first new arm selection in the evaluation window
            try:
                time_to_first_selection = window_selections.index(new_arm_index)
            except ValueError:
                time_to_first_selection = float("inf")

            cold_start_metrics_list.append(
                {
                    "Algorithm": algo_name,
                    "Time to First New Arm Selection": time_to_first_selection,
                    "New Arm Selection Count": np.sum(new_arm_selections),
                    "Exploration Rate": exploration_rate,
                    "Cold-Start AUC": cold_start_auc,
                }
            )

            # Compute cumulative new arm rewards over the evaluation window for plotting
            cumulative_rewards = np.cumsum(
                [
                    reward if arm == new_arm_index else 0
                    for arm, reward in zip(window_selections, window_rewards)
                ]
            )
            cumulative_new_arm_rewards[algo_name] = cumulative_rewards

        # Create DataFrames from the collected metrics
        strategy_distribution_df = pd.DataFrame(strategy_distribution_list)
        cold_start_metrics_df = pd.DataFrame(cold_start_metrics_list)

        # Generate strategy distribution plot: bar plots for each algorithm
        algorithms = strategy_distribution_df["Algorithm"].unique()
        n_algos = len(algorithms)
        fig1, axes = plt.subplots(1, n_algos, figsize=(7, 5 * n_algos), sharey=True)
        if n_algos == 1:
            axes = [axes]
        for ax, algo in zip(axes, algorithms):
            data = strategy_distribution_df[strategy_distribution_df["Algorithm"] == algo]
            ax.bar(data["Arm"], data["Selection Frequency"])
            ax.set_title(f"{algo} Selection Distribution")
            ax.set_xlabel("Arm")
            ax.set_ylabel("Frequency")
        plt.tight_layout()

        # Generate cold start AUC plot: cumulative new arm reward over the evaluation window
        cold_start_time = np.arange(evaluation_period_steps)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for algo, cum_rewards in cumulative_new_arm_rewards.items():
            ax2.plot(cold_start_time, cum_rewards, label=algo)
        ax2.set_xlabel("Time Steps in Evaluation Period")
        ax2.set_ylabel("Cumulative New Arm Reward")
        ax2.set_title("Cold-Start Performance: Cumulative New Arm Reward")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        return {
            "strategy_distribution_table": strategy_distribution_df,
            "strategy_distribution_plot": fig1,
            "cold_start_metrics_table": cold_start_metrics_df,
            "cold_start_auc_plot": fig2,
        }
