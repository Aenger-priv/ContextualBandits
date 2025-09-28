import numpy as np
from bandit_base import ContextualBandit
import pandas as pd
from collections import deque


class LinUCB(ContextualBandit):
    """Linear Upper Confidence Bound algorithm."""

    def __init__(self, n_arms, context_dim, alpha=1.0):
        """
        Initialize LinUCB algorithm.

        Args:
            n_arms: Number of arms
            context_dim: Dimension of the context vector
            alpha: Exploration parameter
        """
        print(f"Initializing LinUCB with alpha={alpha}")  # Debug log
        super().__init__(n_arms, context_dim)
        self.name = f"LinUCB (alpha={alpha})"
        self.alpha = alpha

        # Initialize model parameters for each arm
        self.A = [np.identity(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        self.theta = [np.zeros(context_dim) for _ in range(n_arms)]

        # Initialize context processor
        self.context_processor = None

    def preprocess_context(self, context):
        """Convert context dictionary to feature vector."""
        if self.context_processor is None:
            raise ValueError("Context processor not set")
        return self.context_processor.transform(context)

    def add_arm(self):
        """Add a new arm and initialize its parameters."""
        super().add_arm()
        self.A.append(np.identity(self.context_dim))
        self.b.append(np.zeros(self.context_dim))
        self.theta.append(np.zeros(self.context_dim))

    def select_arm(self, context):
        """
        Select an arm using LinUCB algorithm.

        Args:
            context: Context dictionary

        Returns:
            arm: Index of the selected arm
        """
        context_vector = self.preprocess_context(context)

        # Calculate UCB for each arm
        ucb_values = []
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv.dot(self.b[arm])
            self.theta[arm] = theta

            # Calculate UCB
            cb = self.alpha * np.sqrt(context_vector.dot(A_inv).dot(context_vector))
            ucb = context_vector.dot(theta) + cb
            ucb_values.append(ucb)

        # Select arm with highest UCB
        selected_arm = np.argmax(ucb_values)
        self.selected_arms.append(selected_arm)
        return selected_arm

    def update(self, context, arm, reward):
        """
        Update model parameters based on observed context, arm, and reward.

        Args:
            context: Context dictionary
            arm: Selected arm
            reward: Observed reward
        """
        self.t += 1
        self.obtained_rewards.append(reward)
        self.cumulative_reward += reward

        # Update model parameters for the selected arm
        context_vector = self.preprocess_context(context)
        self.A[arm] += np.outer(context_vector, context_vector)
        self.b[arm] += reward * context_vector


class SlidingWindowLinUCB(LinUCB):
    """Sliding Window Linear Upper Confidence Bound algorithm."""

    def __init__(self, n_arms, context_dim, alpha=1.0, window_size=100):
        """
        Initialize Sliding Window LinUCB algorithm.

        Args:
            n_arms: Number of arms
            context_dim: Dimension of the context vector
            alpha: Exploration parameter
            window_size: Size of the sliding window
        """
        print(f"Initializing SW-LinUCB with alpha={alpha}, window_size={window_size}")  # Debug log
        super().__init__(n_arms, context_dim, alpha)
        self.name = f"SW-LinUCB (alpha={alpha} w={window_size})"
        self.window_size = window_size

        # Store history for sliding window
        self.context_history = [[] for _ in range(n_arms)]
        self.reward_history = [[] for _ in range(n_arms)]

    def add_arm(self):
        """Add a new arm and initialize its parameters."""
        super().add_arm()
        self.A.append(np.identity(self.context_dim))
        self.b.append(np.zeros(self.context_dim))
        self.theta.append(np.zeros(self.context_dim))
        # Add history arrays for the new arm
        self.context_history.append([])
        self.reward_history.append([])

    def update(self, context, arm, reward):
        """
        Update model with sliding window approach.

        Args:
            context: Context dictionary
            arm: Selected arm
            reward: Observed reward
        """
        self.t += 1
        self.obtained_rewards.append(reward)
        self.cumulative_reward += reward

        # Add new observation to history
        context_vector = self.preprocess_context(context)
        self.context_history[arm].append(context_vector)
        self.reward_history[arm].append(reward)

        # Remove old observations if window is full
        if len(self.context_history[arm]) > self.window_size:
            self.context_history[arm].pop(0)
            self.reward_history[arm].pop(0)

        # Recalculate A and b matrices for the selected arm
        self.A[arm] = np.identity(self.context_dim)
        self.b[arm] = np.zeros(self.context_dim)

        for c, r in zip(self.context_history[arm], self.reward_history[arm]):
            self.A[arm] += np.outer(c, c)
            self.b[arm] += r * c


class LinUCBDecay(ContextualBandit):
    """LinUCB with exponential decay for non-stationary environments."""

    def __init__(self, n_arms, context_dim, alpha=1.0, decay=0.95):
        """
        Initialize LinUCBDecay algorithm.

        Args:
            n_arms: Number of arms
            context_dim: Dimension of the context vector
            alpha: Exploration parameter
            decay: Decay factor for exponential weighting (0 < decay <= 1)
        """
        print(f"Initializing LinUCBDecay with alpha={alpha}, decay={decay}")
        super().__init__(n_arms, context_dim)
        self.name = f"LinUCBDecay (alpha={alpha}, decay={decay})"
        self.alpha = alpha
        self.decay = decay

        # Model parameters
        self.A = [np.identity(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        self.theta = [np.zeros(context_dim) for _ in range(n_arms)]

        # Context processor placeholder
        self.context_processor = None

    def preprocess_context(self, context):
        """Convert context dictionary to feature vector."""
        if self.context_processor is None:
            raise ValueError("Context processor not set")
        return self.context_processor.transform(context)

    def add_arm(self):
        """Add a new arm and initialize its parameters."""
        super().add_arm()
        self.A.append(np.identity(self.context_dim))
        self.b.append(np.zeros(self.context_dim))
        self.theta.append(np.zeros(self.context_dim))

    def select_arm(self, context):
        """
        Select an arm using LinUCB with decay.

        Args:
            context: Context dictionary

        Returns:
            arm: Index of the selected arm
        """
        context_vector = self.preprocess_context(context)

        ucb_values = []
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv.dot(self.b[arm])
            self.theta[arm] = theta

            cb = self.alpha * np.sqrt(max(context_vector.dot(A_inv).dot(context_vector), 0))
            ucb = context_vector.dot(theta) + cb
            ucb_values.append(ucb)

        selected_arm = np.argmax(ucb_values)
        self.selected_arms.append(selected_arm)
        return selected_arm

    def update(self, context, arm, reward):
        """
        Update model parameters with exponential decay.

        Args:
            context: Context dictionary
            arm: Selected arm
            reward: Observed reward
        """
        self.t += 1
        self.obtained_rewards.append(reward)
        self.cumulative_reward += reward

        context_vector = self.preprocess_context(context)

        # Apply exponential decay to A and b
        self.A[arm] = self.decay * self.A[arm] + np.outer(context_vector, context_vector)
        self.b[arm] = self.decay * self.b[arm] + reward * context_vector


class LinTS(ContextualBandit):
    """Linear Thompson Sampling algorithm."""

    def __init__(self, n_arms, context_dim, v=0.25):
        """
        Initialize Linear Thompson Sampling algorithm.

        Args:
            n_arms: Number of arms
            context_dim: Dimension of the context vector
            v: Exploration parameter (variance)
        """
        print(f"Initializing LinTS with v={v}")  # Debug log
        super().__init__(n_arms, context_dim)
        self.name = f"LinTS (v={v})"
        self.v = v

        # Initialize model parameters for each arm
        self.A = [np.identity(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        self.theta_hat = [np.zeros(context_dim) for _ in range(n_arms)]

        # Initialize context processor
        self.context_processor = None

    def preprocess_context(self, context):
        """Convert context dictionary to feature vector."""
        if self.context_processor is None:
            raise ValueError("Context processor not set")
        return self.context_processor.transform(context)

    def select_arm(self, context):
        """
        Select an arm using Thompson Sampling.

        Args:
            context: Context dictionary

        Returns:
            arm: Index of the selected arm
        """
        context_vector = self.preprocess_context(context)

        # Sample theta for each arm
        sampled_rewards = []
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            mu = A_inv.dot(self.b[arm])
            self.theta_hat[arm] = mu

            # Sample from multivariate normal distribution
            cov = self.v * A_inv
            theta_tilde = np.random.multivariate_normal(mu, cov)

            # Calculate expected reward
            reward = context_vector.dot(theta_tilde)
            sampled_rewards.append(reward)

        # Select arm with highest sampled reward
        selected_arm = np.argmax(sampled_rewards)
        self.selected_arms.append(selected_arm)
        return selected_arm

    def update(self, context, arm, reward):
        """
        Update model parameters based on observed context, arm, and reward.

        Args:
            context: Context dictionary
            arm: Selected arm
            reward: Observed reward
        """
        self.t += 1
        self.obtained_rewards.append(reward)
        self.cumulative_reward += reward

        # Update model parameters for the selected arm
        context_vector = self.preprocess_context(context)
        self.A[arm] += np.outer(context_vector, context_vector)
        self.b[arm] += reward * context_vector

    def add_arm(self):
        """Add a new arm and initialize its parameters."""
        super().add_arm()
        self.A.append(np.identity(self.context_dim))
        self.b.append(np.zeros(self.context_dim))
        self.theta_hat.append(np.zeros(self.context_dim))


class EpsilonGreedy(ContextualBandit):
    """Epsilon-Greedy contextual bandit algorithm."""

    def __init__(self, n_arms, context_dim, epsilon=0.1):
        """
        Initialize Epsilon-Greedy algorithm.

        Args:
            n_arms: Number of arms
            context_dim: Dimension of the context vector
            epsilon: Exploration probability
        """
        print(f"Initializing EpsilonGreedy with epsilon={epsilon}")  # Debug log
        super().__init__(n_arms, context_dim)
        self.name = f"ε-Greedy (ε={epsilon})"
        self.epsilon = epsilon

        # Simple linear model for each arm
        self.W = [np.zeros(context_dim) for _ in range(n_arms)]
        self.n_pulls = np.zeros(n_arms)
        self.sum_rewards = np.zeros(n_arms)

        # Keep track of context-reward pairs for each arm
        self.contexts = [[] for _ in range(n_arms)]
        self.rewards = [[] for _ in range(n_arms)]

        # Initialize context processor
        self.context_processor = None

    def preprocess_context(self, context):
        """Convert context dictionary to feature vector."""
        if self.context_processor is None:
            raise ValueError("Context processor not set")
        return self.context_processor.transform(context)

    def select_arm(self, context):
        """
        Select an arm using Epsilon-Greedy strategy.

        Args:
            context: Context dictionary

        Returns:
            arm: Index of the selected arm
        """
        context_vector = self.preprocess_context(context)

        # Explore with probability epsilon
        if np.random.random() < self.epsilon:
            selected_arm = np.random.randint(self.n_arms)
        else:
            # Exploit: choose arm with highest predicted reward
            predicted_rewards = [
                context_vector.dot(self.W[arm]) if self.n_pulls[arm] > 0 else 0
                for arm in range(self.n_arms)
            ]
            selected_arm = np.argmax(predicted_rewards)

        self.selected_arms.append(selected_arm)
        return selected_arm

    def update(self, context, arm, reward):
        """
        Update model parameters based on observed context, arm, and reward.

        Args:
            context: Context dictionary
            arm: Selected arm
            reward: Observed reward
        """
        self.t += 1
        self.obtained_rewards.append(reward)
        self.cumulative_reward += reward

        # Update statistics for the selected arm
        context_vector = self.preprocess_context(context)
        self.n_pulls[arm] += 1
        self.sum_rewards[arm] += reward

        # Store context-reward pair
        self.contexts[arm].append(context_vector)
        self.rewards[arm].append(reward)

        # Update linear model for this arm using ridge regression
        X = np.array(self.contexts[arm])
        y = np.array(self.rewards[arm])

        # Simple online update
        if len(X) >= 5:  # Only update after collecting enough samples
            # Use ridge regression with small regularization
            ridge_lambda = 0.1
            A = X.T.dot(X) + ridge_lambda * np.eye(self.context_dim)
            b = X.T.dot(y)
            try:
                self.W[arm] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular
                self.W[arm] = np.linalg.lstsq(X, y, rcond=None)[0]

    def add_arm(self):
        """Add a new arm and initialize its parameters."""
        super().add_arm()
        # Add new parameters for the new arm
        self.W.append(np.zeros(self.context_dim))
        self.n_pulls = np.append(self.n_pulls, 0)
        self.sum_rewards = np.append(self.sum_rewards, 0)
        self.contexts.append([])
        self.rewards.append([])


class SlidingDoublyRobustSoftmax(ContextualBandit):
    """Sliding Doubly Robust Softmax bandit algorithm."""

    def __init__(self, n_arms, context_dim, tau=0.1, window_size=100, lambda_reg=0.1):
        """
        Initialize Sliding Doubly Robust Softmax algorithm.

        Args:
            n_arms: Number of arms
            context_dim: Dimension of the context vector
            tau: Temperature parameter for Softmax
            window_size: Size of the sliding window
            lambda_reg: Regularization parameter for ridge regression
        """
        print(f"Initializing DR-Softmax with tau={tau}, window_size={window_size}")  # Debug log
        super().__init__(n_arms, context_dim)
        self.name = f"DR-Softmax (τ={tau}, w={window_size})"
        self.tau = tau
        self.window_size = window_size
        self.lambda_reg = lambda_reg

        # For each arm, maintain a linear model
        self.theta = [np.zeros(context_dim) for _ in range(n_arms)]

        # Use deque with maxlen for automatic sliding window
        self.context_history = deque(maxlen=window_size)
        self.action_history = deque(maxlen=window_size)
        self.reward_history = deque(maxlen=window_size)
        self.prob_history = deque(maxlen=window_size)

        # Initialize context processor
        self.context_processor = None

        # initialize update and selection logs
        self.selection_log_df = pd.DataFrame(
            columns=[
                "step",
                "context",
                "estimates",
                "selected_arm",
                "decision_type",
                "tau",
                "window_size",
            ]
        )
        self.update_log_df = pd.DataFrame(columns=["step", "observed_reward", "dr_estimates"])

    def preprocess_context(self, context):
        """Convert context dictionary to feature vector."""
        if self.context_processor is None:
            raise ValueError("Context processor not set")
        return self.context_processor.transform(context)

    def log_selection(self, context, estimates, selected_arm, decision_type):
        """Log the details at the select_arm stage.

        Args:
            context: The context dictionary for the current step.
            estimates: List of per-arm estimates (scores).
            selected_arm: The index of the selected arm.
            decision_type: 'exploring' if not the maximum estimate, otherwise 'exploiting'.
        """
        step = self.t  # using the current time step from the parent class
        new_entry = {
            "step": step,
            "context": context,
            "estimates": estimates,
            "selected_arm": selected_arm,
            "decision_type": decision_type,
            "tau": self.tau,
            "window_size": self.window_size,
        }
        self.selection_log_df.loc[len(self.selection_log_df)] = new_entry

    def ensure_theta_length(self):
        """Ensure theta array has entries for all arms by extending with zero vectors if needed."""

        if len(self.theta) < self.n_arms:
            extra = self.n_arms - len(self.theta)
            self.theta.extend([np.zeros(self.context_dim) for _ in range(extra)])

    def select_arm(self, context):
        """
        Select an arm using Softmax policy.

        Args:
            context: Context dictionary

        Returns:
            arm: Index of the selected arm
        """
        # Ensure that self.theta has entries for all arms
        self.ensure_theta_length()
        context_vector = self.preprocess_context(context)

        # Calculate scores for each arm
        scores = np.array([context_vector.dot(self.theta[arm]) for arm in range(self.n_arms)])

        # Apply Softmax with temperature
        max_score = np.max(scores)
        exp_scores = np.exp((scores - max_score) / self.tau)
        probs = exp_scores / np.sum(exp_scores)

        # Sample arm according to probabilities
        selected_arm = np.random.choice(self.n_arms, p=probs)
        self.selected_arms.append(selected_arm)

        # Determine if the decision is exploring or exploiting
        if selected_arm == np.argmax(scores):
            decision_type = "exploiting"
        else:
            decision_type = "exploring"

        # Log the selection details
        self.log_selection(context, scores.tolist(), selected_arm, decision_type)

        # Store selection probabilities for DR estimation
        self._current_probs = probs

        return selected_arm

    def update(self, context, arm, reward):
        """
        Update model using Doubly Robust estimation.

        Args:
            context: Context dictionary
            arm: Selected arm
            reward: Observed reward
        """
        self.t += 1
        self.obtained_rewards.append(reward)
        self.cumulative_reward += reward

        context_vector = self.preprocess_context(context)

        # Store data in history (deque will automatically handle window size)
        self.context_history.append(context_vector)
        self.action_history.append(arm)
        self.reward_history.append(reward)
        self.prob_history.append(self._current_probs)

        # Update reward models for all arms
        X = np.array(self.context_history)

        for a in range(self.n_arms):
            # Get indices where this arm was selected
            indices = [i for i, action in enumerate(self.action_history) if action == a]

            if len(indices) > 0:
                # Extract contexts and rewards for this arm
                X_arm = X[indices]
                y_arm = np.array([self.reward_history[i] for i in indices])

                # Ridge regression with regularization
                A = X_arm.T.dot(X_arm) + self.lambda_reg * np.eye(self.context_dim)
                b = X_arm.T.dot(y_arm)

                try:
                    self.theta[a] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    # Fallback if matrix is singular
                    self.theta[a] = np.linalg.lstsq(X_arm, y_arm, rcond=None)[0]

        # Compute DR estimates and update models
        if len(self.context_history) > 10:  # Only start DR updates after collecting some data
            self._update_dr()

        # After updating models, compute DR estimates for non-chosen arms
        dr_estimates = {}
        for a in range(self.n_arms):
            if a != arm:
                # For non-chosen arms, the DR estimate is simply the predicted reward
                dr_estimates[a] = context_vector.dot(self.theta[a])
        self.log_update(reward, dr_estimates)

    def log_update(self, observed_reward, dr_estimates):
        """
        Log update details for the current step, including the observed reward and DR estimates for non-chosen arms.

        Args:
            observed_reward: The reward observed for the chosen arm.
            dr_estimates: A dictionary mapping arm indices (for non-chosen arms) to their DR estimates.
        """
        step = self.t  # using the current time step
        new_entry = {
            "step": step,
            "observed_reward": observed_reward,
            "dr_estimates": dr_estimates,
        }
        self.update_log_df.loc[len(self.update_log_df)] = new_entry

    def _update_dr(self):
        """Update models using Doubly Robust estimation."""
        X = np.array(self.context_history)

        # For each arm, compute DR estimates
        for a in range(self.n_arms):
            dr_rewards = []
            dr_contexts = []

            for i in range(len(self.context_history)):
                context = self.context_history[i]
                action = self.action_history[i]
                reward = self.reward_history[i]
                probs = self.prob_history[i]

                # Predict reward using current model
                predicted_reward = context.dot(self.theta[a])

                # Compute DR estimate
                if action == a:
                    # Using the conventional DR formula:
                    dr_reward = predicted_reward + (reward - predicted_reward) / probs[a]
                else:
                    # Use model prediction
                    dr_reward = predicted_reward

                dr_rewards.append(dr_reward)
                dr_contexts.append(context)

            # Update model with DR estimates
            dr_X = np.array(dr_contexts)
            dr_y = np.array(dr_rewards)

            # Ridge regression with regularization
            A = dr_X.T.dot(dr_X) + self.lambda_reg * np.eye(self.context_dim)
            b = dr_X.T.dot(dr_y)

            try:
                self.theta[a] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular
                self.theta[a] = np.linalg.lstsq(dr_X, dr_y, rcond=None)[0]

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


class RidgeSoftmax(ContextualBandit):
    """Plain Softmax bandit algorithm without doubly robust estimates or sliding window."""

    def __init__(self, n_arms, context_dim, tau=0.1, lambda_reg=0.1):
        """
        Initialize Plain Softmax algorithm.

        Args:
            n_arms: Number of arms.
            context_dim: Dimension of the context vector.
            tau: Temperature parameter for Softmax.
            lambda_reg: Regularization parameter for ridge regression.
        """
        print(f"Initializing Plain Softmax with tau={tau}")
        super().__init__(n_arms, context_dim)
        self.name = f"Plain Softmax (τ={tau})"
        self.tau = tau
        self.lambda_reg = lambda_reg

        # For each arm, maintain a linear model parameter vector
        self.theta = [np.zeros(context_dim) for _ in range(n_arms)]

        # History of all observations (using lists, not a sliding window)
        self.context_history = []
        self.action_history = []
        self.reward_history = []

        # Context processor will be set externally
        self.context_processor = None

        # Initialize logs
        self.selection_log_df = pd.DataFrame(
            columns=["step", "context", "estimates", "selected_arm", "decision_type", "tau"]
        )
        self.update_log_df = pd.DataFrame(columns=["step", "observed_reward"])

    def preprocess_context(self, context):
        """Convert context dictionary to feature vector using the context processor."""
        if self.context_processor is None:
            raise ValueError("Context processor not set")
        return self.context_processor.transform(context)

    def log_selection(self, context, estimates, selected_arm, decision_type):
        """Log details at the arm selection stage."""
        step = self.t  # using current time step from the parent class
        new_entry = {
            "step": step,
            "context": context,
            "estimates": estimates,
            "selected_arm": selected_arm,
            "decision_type": decision_type,
            "tau": self.tau,
        }
        self.selection_log_df.loc[len(self.selection_log_df)] = new_entry

    def ensure_theta_length(self):
        """Ensure theta has an entry for each arm."""
        if len(self.theta) < self.n_arms:
            extra = self.n_arms - len(self.theta)
            self.theta.extend([np.zeros(self.context_dim) for _ in range(extra)])

    def select_arm(self, context):
        """
        Select an arm using a Softmax policy.

        Args:
            context: Context dictionary.

        Returns:
            selected_arm: Index of the selected arm.
        """
        self.ensure_theta_length()
        context_vector = self.preprocess_context(context)

        # Calculate scores for each arm based on current parameter estimates
        scores = np.array([context_vector.dot(self.theta[a]) for a in range(self.n_arms)])

        # Compute Softmax probabilities using the temperature parameter tau
        exp_scores = np.exp(scores / self.tau)
        probs = exp_scores / np.sum(exp_scores)

        # Sample an arm according to these probabilities
        selected_arm = np.random.choice(self.n_arms, p=probs)
        self.selected_arms.append(selected_arm)

        # Determine if the selection is exploiting or exploring
        decision_type = "exploiting" if selected_arm == np.argmax(scores) else "exploring"
        self.log_selection(context, scores.tolist(), selected_arm, decision_type)

        # Optionally store current probabilities (not used further here)
        self._current_probs = probs

        return selected_arm

    def update(self, context, arm, reward):
        """
        Update model parameters based on the observed reward.

        Args:
            context: Context dictionary.
            arm: Selected arm.
            reward: Observed reward.
        """
        # Update time step and reward tracking (assuming these are initialized in the parent class)
        self.t += 1
        self.obtained_rewards.append(reward)
        self.cumulative_reward += reward

        context_vector = self.preprocess_context(context)

        # Append current observation to history
        self.context_history.append(context_vector)
        self.action_history.append(arm)
        self.reward_history.append(reward)

        # Update reward model parameters for each arm using all historical data
        X = np.array(self.context_history)
        for a in range(self.n_arms):
            # Find indices where arm 'a' was selected
            indices = [i for i, action in enumerate(self.action_history) if action == a]
            if indices:
                X_a = X[indices]
                y_a = np.array([self.reward_history[i] for i in indices])
                # Ridge regression update (regularized least squares)
                A = X_a.T.dot(X_a) + self.lambda_reg * np.eye(self.context_dim)
                b = X_a.T.dot(y_a)
                try:
                    self.theta[a] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    # Fallback to least squares solution if singular
                    self.theta[a] = np.linalg.lstsq(X_a, y_a, rcond=None)[0]

        self.log_update(reward)

    def log_update(self, observed_reward):
        """Log update details for the current step."""
        step = self.t  # using current time step
        new_entry = {
            "step": step,
            "observed_reward": observed_reward,
        }
        self.update_log_df.loc[len(self.update_log_df)] = new_entry

    def add_arm(self):
        """Add a new arm and initialize its parameters."""
        super().add_arm()
        self.theta.append(np.zeros(self.context_dim))
