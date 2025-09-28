from datetime import datetime, timedelta
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FXDatasetGenerator:
    """
    Generates synthetic FX trading data for contextual bandit algorithm research.
    Simulates non-stationary reward distributions, imbalanced contexts,
    paradigm shifts, and cold starts with new strategies.
    """

    def __init__(
        self,
        currency_pairs: List[str] = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"],
        start_date: str = "2023-01-01",
        end_date: str = "2023-12-31",
        n_arms: int = 5,
        new_arm_day: int = 180,  # Day to introduce a new arm (cold start simulation)
        regime_change_days: List[int] = [90, 240],  # Days when market regimes change
        time_of_day_categories: List[str] = ["morning", "midday", "afternoon", "pre-close"],
        random_seed: int = 42,
    ):
        """
        Initialize the FX dataset generator.

        Args:
            currency_pairs: List of currency pairs to simulate
            start_date: Start date for simulation
            end_date: End date for simulation
            n_arms: Number of trading strategies (arms)
            new_arm_day: Day to introduce a new arm
            regime_change_days: Days when market regimes change (paradigm shifts)
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        self.currency_pairs = currency_pairs
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.n_days = (self.end_date - self.start_date).days + 1
        self.dates = [self.start_date + timedelta(days=i) for i in range(self.n_days)]
        self.n_arms = n_arms
        self.new_arm_day = new_arm_day
        self.regime_change_days = regime_change_days
        self.time_of_day_categories = time_of_day_categories

        # Size imbalance parameters (some contexts will appear more frequently)
        self.size_params = {
            "EUR/USD": {"mean": 1000, "std": 200, "weight": 0.3},
            "GBP/USD": {"mean": 500, "std": 100, "weight": 0.2},
            "USD/JPY": {"mean": 750, "std": 150, "weight": 0.2},
            "AUD/USD": {"mean": 300, "std": 50, "weight": 0.1},
            "USD/CAD": {"mean": 250, "std": 50, "weight": 0.2},
        }

        # Create volatility time series first
        self.generate_volatility_timeseries()

        # Initialize arm reward parameters
        self.init_arm_reward_parameters()

    def generate_volatility_timeseries(self):
        """
        Generate consistent volatility time series for each currency pair.
        Uses regime changes to create non-stationarity in volatility.
        """
        # Base volatility for each currency pair
        base_volatility = {
            "EUR/USD": 0.08,
            "GBP/USD": 0.12,
            "USD/JPY": 0.10,
            "AUD/USD": 0.15,
            "USD/CAD": 0.11,
        }

        # Create empty DataFrame for volatility time series
        self.volatility = pd.DataFrame(index=self.dates)

        # Generate volatility for each currency pair
        for pair in self.currency_pairs:
            # Start with base volatility
            vol = np.zeros(self.n_days)

            # Add regime changes
            regimes = [0] + self.regime_change_days + [self.n_days]

            for i in range(len(regimes) - 1):
                start_idx = regimes[i]
                end_idx = regimes[i + 1]

                # Different volatility levels for different regimes
                regime_multiplier = 0.8 + (i % 3) * 0.4  # Cycles through 0.8, 1.2, 1.6
                base_vol = base_volatility[pair] * regime_multiplier

                # GARCH-like volatility process
                for j in range(start_idx, end_idx):
                    if j == start_idx:
                        vol[j] = base_vol
                    else:
                        # AR(1) process with mean reversion
                        vol[j] = 0.95 * vol[j - 1] + 0.05 * base_vol + np.random.normal(0, 0.01)
                        vol[j] = max(0.03, vol[j])  # Ensure positive volatility

            # Add seasonality pattern (e.g., higher volatility on Mondays)
            for j in range(self.n_days):
                day_of_week = self.dates[j].weekday()
                if day_of_week == 0:  # Monday
                    vol[j] *= 1.1
                elif day_of_week == 4:  # Friday
                    vol[j] *= 1.05

            # Add volatility spikes (simulating news events)
            n_spikes = int(self.n_days * 0.03)  # 3% of days have spikes
            spike_indices = np.random.choice(range(self.n_days), size=n_spikes, replace=False)
            for idx in spike_indices:
                vol[idx] *= 1.5 + np.random.rand()

            self.volatility[pair] = vol

    def init_arm_reward_parameters(self):
        """
        Initialize parameters for how each arm (strategy) performs in different contexts.
        Strategies will have different effectiveness based on:
        - Currency pair
        - Volatility level
        - Size of the trade
        - Time of day
        """
        # Strategy characteristics
        # Each strategy will have strengths and weaknesses in different contexts

        # Rows: currency pairs, Columns: arms (strategies)
        self.currency_effectiveness = {}

        # Strategy 0: Trend following - good for trending pairs, bad in ranges
        # Strategy 1: Mean reversion - good for rangebound pairs
        # Strategy 2: Breakout strategy - good for volatile pairs
        # Strategy 3: Carry trade strategy - good for certain pairs
        # Strategy 4: Technical pattern - performs differently across pairs
        # Strategy 5: (Will be introduced later) - Adaptive strategy

        # Each strategy's effectiveness per currency pair (base performance)
        for i, pair in enumerate(self.currency_pairs):
            # Each currency pair has a different profile for strategy effectiveness
            if pair == "EUR/USD":
                self.currency_effectiveness[pair] = [0.6, 0.3, 0.7, 0.2, 0.7]
            elif pair == "GBP/USD":
                self.currency_effectiveness[pair] = [0.5, 0.6, 0.7, 0.3, 0.4]
            elif pair == "USD/JPY":
                self.currency_effectiveness[pair] = [0.3, 0.4, 0.6, 0.7, 0.9]
            elif pair == "AUD/USD":
                self.currency_effectiveness[pair] = [0.4, 0.6, 0.5, 0.8, 0.3]
            elif pair == "USD/CAD":
                self.currency_effectiveness[pair] = [0.1, 0.5, 0.5, 0.5, 0.6]

        # Volatility impact on each strategy
        # [slope, optimal_volatility, width]
        # Slope: how much volatility affects the strategy (positive or negative)
        # Optimal_volatility: at what volatility level the strategy performs best
        # Width: how sensitive the strategy is to deviations from optimal volatility
        self.volatility_impact = [
            [1.0, 0.12, 0.1],  # Double the slope from 0.5 to 1.0
            [-1.5, 0.06, 0.08],  # Increase negative impact
            [2.0, 0.20, 0.15],  # Stronger high volatility preference
            [-1.0, 0.08, 0.06],  # Stronger low volatility preference
            [0.8, 0.15, 0.12],  # Increased medium-high volatility preference
        ]

        # Size impact on each strategy (some strategies work better for larger sizes)
        # [small_size_effectiveness, large_size_effectiveness]
        self.size_impact = [
            [0.9, 0.3],  # Much better for small sizes
            [0.3, 0.9],  # Much better for large sizes
            [0.2, 1.0],  # Extremely size-dependent
            [1.0, 0.2],  # Extremely size-dependent (opposite)
            [0.6, 0.6],  # Keep one neutral strategy
        ]

        # Time of day impact on each strategy
        # Each row represents a strategy, each column a time of day category
        # Format: [morning, midday, afternoon, pre-close]
        self.time_of_day_impact = {
            0: [1.2, 0.9, 0.8, 0.7],  # Strategy 0: Strong in morning, weakens throughout day
            1: [0.8, 1.0, 1.1, 0.9],  # Strategy 1: Best mid-day and afternoon
            2: [0.7, 0.8, 1.0, 1.3],  # Strategy 2: Strongest pre-close
            3: [1.2, 0.9, 0.2, 0.1],  # Strategy 3: Best in morning
            4: [0.9, 1.0, 1.1, 0.8],  # Strategy 4: Best in afternoon
        }

    def get_reward_distribution(
        self, currency_pair: str, volatility: float, size: float, time_of_day: str, day: int
    ) -> Dict[int, float]:
        """
        Calculate the reward distribution for each arm given the context.

        Args:
            currency_pair: The currency pair to trade
            volatility: Current volatility level
            size: Size of the trade
            time_of_day: Time of day category ("morning", "midday", "afternoon", "pre-close")
            day: Current day in the simulation

        Returns:
            Dictionary mapping arm indices to expected rewards
        """
        rewards = {}

        # Determine if we've introduced the new arm yet
        active_arms = self.n_arms
        if day < self.new_arm_day:
            active_arms = self.n_arms - 1

        # Size factor (normalized between 0 and 1)
        size_factor = min(1.0, size / 1500)

        # Calculate expected reward for each active arm
        for arm in range(active_arms):
            # Base reward from currency pair
            base_reward = self.currency_effectiveness[currency_pair][arm]

            # Adjust for volatility (quadratic function centered at optimal volatility)
            vol_params = self.volatility_impact[arm]
            vol_factor = 1.0 - ((volatility - vol_params[1]) / vol_params[2]) ** 2
            vol_factor = max(0.2, min(1.0, vol_factor))
            vol_effect = 1.0 + vol_params[0] * (vol_factor - 0.5)

            # Adjust for size
            small_eff, large_eff = self.size_impact[arm]
            size_effect = small_eff * (1 - size_factor) + large_eff * size_factor

            # Combined effect
            reward = base_reward * vol_effect * size_effect

            # Apply regime changes (paradigm shifts)
            regime_idx = 0
            for i, change_day in enumerate(self.regime_change_days):
                if day >= change_day:
                    regime_idx = i + 1

            # Different regimes favor different strategies
            regime_effect = 1.0 + 0.2 * np.sin(arm + regime_idx)

            reward *= regime_effect

            # Bound the reward between 0 and 1
            reward = max(0.0, min(1.0, reward))

            # Round the reward to 5 decimals
            reward = round(reward, 5)

            rewards[arm] = reward

        # If the new arm is active, add it
        if day >= self.new_arm_day and self.n_arms > active_arms:
            # New adaptive strategy (arm 5) - starts with uncertain performance
            # but improves over time as it "learns"
            days_since_intro = day - self.new_arm_day
            learning_curve = min(1.0, days_since_intro / 60)  # Ramps up over 60 days

            # Combines aspects of other strategies, adapting to the context
            adaptive_reward = (
                0.3
                + 0.4 * learning_curve
                + 0.2 * np.sin(2 * np.pi * day / 30)  # Cyclical performance pattern
            )

            rewards[active_arms] = adaptive_reward

        return rewards

    def generate_context_batch(self, current_day: int, batch_size: int = 100) -> List[Dict]:
        """
        Generate a batch of context instances for a given day.

        Args:
            current_day: Current day in the simulation
            batch_size: Number of context instances to generate

        Returns:
            List of context dictionaries
        """
        date = self.dates[current_day]
        contexts = []

        # Sample currency pairs based on weights (creates imbalance)
        weights = [self.size_params[pair]["weight"] for pair in self.currency_pairs]
        selected_pairs = np.random.choice(self.currency_pairs, size=batch_size, p=weights)

        for pair in selected_pairs:
            # Get current volatility from time series
            volatility = self.volatility.loc[date, pair]

            # Sample size based on currency pair parameters (mean and std)
            size_params = self.size_params[pair]
            size = max(10, int(np.random.normal(size_params["mean"], size_params["std"])))

            # Sample time of day with different distributions based on the currency pair
            # This creates realistic patterns like EUR/USD being more active during European/US hours
            if pair in ["EUR/USD", "GBP/USD"]:
                # European pairs have more activity in morning/midday (European hours)
                time_weights = [0.3, 0.3, 0.25, 0.15]
            elif pair in ["USD/JPY", "AUD/USD"]:
                # Asian pairs have more activity in afternoon/pre-close (Asian hours)
                time_weights = [0.15, 0.25, 0.3, 0.3]
            else:
                # Other pairs have more balanced distribution
                time_weights = [0.25, 0.25, 0.25, 0.25]

            time_of_day = np.random.choice(self.time_of_day_categories, p=time_weights)

            # Get reward distribution for this context
            reward_dist = self.get_reward_distribution(
                pair, volatility, size, time_of_day, current_day
            )

            # Create context dictionary
            context = {
                "context": {
                    "currency_pair": pair,
                    "volatility": round(volatility, 4),
                    "size": size,
                    "time_of_day": time_of_day,
                    "date": date.strftime("%Y-%m-%d"),
                },
                "rewards": reward_dist,
            }

            contexts.append(context)

        return contexts

    def generate_dataset(self, contexts_per_day: int = 100) -> List[Dict]:
        """
        Generate the complete dataset.

        Args:
            contexts_per_day: Number of context instances per day

        Returns:
            List of context dictionaries
        """
        all_contexts = []

        for day in range(self.n_days):
            daily_contexts = self.generate_context_batch(day, contexts_per_day)
            all_contexts.extend(daily_contexts)

        return all_contexts

    def save_dataset(self, filename: str, contexts: List[Dict]) -> None:
        """
        Save the dataset to a JSON file.

        Args:
            filename: Output filename
            contexts: List of context dictionaries
        """
        with open(filename, "w") as f:
            json.dump(contexts, f, indent=2)

    def save_volatility_series(self, filename: str) -> None:
        """
        Save the volatility time series to a CSV file.

        Args:
            filename: Output filename
        """
        self.volatility.to_csv(filename)

    def plot_volatility_series(self) -> None:
        """Plot the volatility time series for each currency pair"""
        plt.figure(figsize=(12, 6))

        # Convert dates to numerical values for plotting
        dates_num = np.arange(len(self.dates))

        for pair in self.currency_pairs:
            plt.plot(dates_num, self.volatility[pair], label=pair)

        # Add vertical lines at regime change points
        for day in self.regime_change_days:
            plt.axvline(x=day, color="gray", linestyle="--", alpha=0.7)

        # Add vertical line for new arm introduction
        plt.axvline(
            x=self.new_arm_day,
            color="red",
            linestyle=":",
            alpha=0.7,
            label="New Strategy Introduction",
        )

        plt.title("Volatility Time Series")
        plt.xlabel("Days")
        plt.ylabel("Volatility")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_arm_effectiveness(self) -> None:
        """Plot the effectiveness of each arm across different contexts"""
        # Sample points across volatility range
        vol_range = np.linspace(0.03, 0.3, 100)
        plt.figure(figsize=(15, 15))

        # Plot for a specific currency pair and size
        pair = "EUR/USD"
        size = 500
        time_of_day = "morning"
        day = 150  # Before new arm

        plt.subplot(3, 1, 1)
        for arm in range(self.n_arms - 1):  # Exclude new arm
            rewards = []
            for vol in vol_range:
                reward_dist = self.get_reward_distribution(pair, vol, size, time_of_day, day)
                rewards.append(reward_dist[arm])
            plt.plot(vol_range, rewards, label=f"Strategy {arm}")

        plt.title(
            f"Strategy Effectiveness vs Volatility ({pair}, Size={size}, Time={time_of_day}, Day={day})"
        )
        plt.xlabel("Volatility")
        plt.ylabel("Expected Reward")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot after new arm introduction
        day = 300  # After new arm
        plt.subplot(3, 1, 2)
        for arm in range(self.n_arms):
            rewards = []
            for vol in vol_range:
                reward_dist = self.get_reward_distribution(pair, vol, size, time_of_day, day)
                if arm in reward_dist:
                    rewards.append(reward_dist[arm])
                else:
                    rewards.append(0)
            plt.plot(vol_range, rewards, label=f"Strategy {arm}")

        plt.title(
            f"Strategy Effectiveness vs Volatility ({pair}, Size={size}, Time={time_of_day}, Day={day})"
        )
        plt.xlabel("Volatility")
        plt.ylabel("Expected Reward")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot effectiveness across time of day
        plt.subplot(3, 1, 3)
        times = self.time_of_day_categories
        volatility = 0.12  # Medium volatility

        for arm in range(self.n_arms - 1):  # Exclude new arm for clarity
            rewards = []
            for tod in times:
                reward_dist = self.get_reward_distribution(pair, volatility, size, tod, day)
                rewards.append(reward_dist[arm])
            plt.bar(
                [i + arm * 0.15 for i in range(len(times))],
                rewards,
                width=0.15,
                label=f"Strategy {arm}",
            )

        plt.title(
            f"Strategy Effectiveness vs Time of Day ({pair}, Size={size}, Volatility={volatility:.2f})"
        )
        plt.xlabel("Time of Day")
        plt.ylabel("Expected Reward")
        plt.xticks([i + 0.3 for i in range(len(times))], times)
        plt.grid(True, alpha=0.3, axis="y")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_currency_distribution(self, dataset: List[Dict]) -> None:
        """
        Plot the distribution of currency pairs in the dataset.

        Args:
            dataset: List of context dictionaries generated by generate_dataset
        """
        # Count occurrences of each currency pair
        currency_counts: Dict[str, int] = {}
        for context in dataset:
            pair = context["context"]["currency_pair"]
            currency_counts[pair] = currency_counts.get(pair, 0) + 1

        # Create bar plot
        plt.figure(figsize=(10, 6))
        pairs = list(currency_counts.keys())
        counts = list(currency_counts.values())

        bars = plt.bar(pairs, counts)

        # Add percentage labels on top of each bar
        total = sum(counts)
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height/total*100:.1f}%",
                ha="center",
                va="bottom",
            )

        plt.title("Distribution of Currency Pairs in Dataset")
        plt.xlabel("Currency Pair")
        plt.ylabel("Number of Instances")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()

    def plot_reward_nonstationarity(self, fixed_context: Dict = None) -> None:
        """
        Plot the non-stationarity of rewards over time for a fixed context.
        If no context is provided, uses a default context.

        Args:
            fixed_context: Optional dictionary with context parameters
        """
        if fixed_context is None:
            fixed_context = {"currency_pair": "EUR/USD", "size": 500, "time_of_day": "morning"}

        # Create time series of rewards for each arm
        days = range(self.n_days)
        rewards_over_time = {arm: [] for arm in range(self.n_arms)}

        for day in days:
            date = self.dates[day]
            volatility = self.volatility.loc[date, fixed_context["currency_pair"]]

            # Get reward distribution for this context
            reward_dist = self.get_reward_distribution(
                fixed_context["currency_pair"],
                volatility,
                fixed_context["size"],
                fixed_context["time_of_day"],
                day,
            )

            # Store rewards for each arm
            for arm in range(self.n_arms):
                if arm in reward_dist:
                    rewards_over_time[arm].append(reward_dist[arm])
                else:
                    rewards_over_time[arm].append(None)  # For arms not yet introduced

        # Plot rewards over time
        plt.figure(figsize=(15, 8))

        for arm in range(self.n_arms):
            rewards = rewards_over_time[arm]
            valid_days = [d for d, r in zip(days, rewards) if r is not None]
            valid_rewards = [r for r in rewards if r is not None]

            plt.plot(valid_days, valid_rewards, label=f"Strategy {arm}")

        # Add vertical lines for regime changes
        for day in self.regime_change_days:
            plt.axvline(
                x=day,
                color="gray",
                linestyle="--",
                alpha=0.7,
                label="Regime Change" if day == self.regime_change_days[0] else None,
            )

        # Add vertical line for new arm introduction
        plt.axvline(
            x=self.new_arm_day,
            color="red",
            linestyle=":",
            alpha=0.7,
            label="New Strategy Introduction",
        )

        plt.title(f"Reward Non-stationarity Over Time\nContext: {fixed_context}")
        plt.xlabel("Days")
        plt.ylabel("Expected Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def add_arm(self):
        """Add a new arm and initialize its parameters."""
        super().add_arm()
        # Add new parameters for the new arm
        self.theta.append(np.zeros(self.context_dim))

        # Update probabilities for existing history
        if self.prob_history:
            for i in range(len(self.prob_history)):
                old_probs = self.prob_history[i]
                new_probs = np.zeros(len(old_probs) + 1)
                new_probs[:-1] = old_probs
                new_probs[-1] = 0  # Initialize probability for new arm as 0
                self.prob_history[i] = new_probs


# Example usage
if __name__ == "__main__":
    # Create dataset generator
    generator = FXDatasetGenerator(
        start_date="2023-01-01",
        end_date="2023-12-31",
        n_arms=5,
        new_arm_day=180,
        regime_change_days=[90, 240],
    )

    # Generate the dataset
    dataset = generator.generate_dataset(contexts_per_day=50)

    # Save the dataset
    generator.save_dataset("fx_trading_dataset.json", dataset)

    # Save the volatility series
    generator.save_volatility_series("volatility_series.csv")

    # Plot the volatility series
    generator.plot_volatility_series()

    # Plot arm effectiveness
    generator.plot_arm_effectiveness()

    # Plot currency distribution
    generator.plot_currency_distribution(dataset)

    # Plot reward non-stationarity
    generator.plot_reward_nonstationarity()

    # Print some sample data
    print(f"Generated {len(dataset)} context instances")
    print("\nSample data:")
    for i in range(3):
        print(json.dumps(dataset[i], indent=2))
