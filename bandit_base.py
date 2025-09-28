import numpy as np
from sklearn.preprocessing import OneHotEncoder


class ContextualBandit:
    """Base class for all contextual bandit algorithms."""

    def __init__(self, n_arms, context_dim):
        """
        Initialize the contextual bandit.

        Args:
            n_arms: Number of arms (strategies)
            context_dim: Dimension of the context vector after preprocessing
        """
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.name = "Base Contextual Bandit"

        # Performance tracking
        self.selected_arms = []
        self.obtained_rewards = []
        self.cumulative_reward = 0
        self.cumulative_regret = 0
        self.t = 0  # Time step counter

    def preprocess_context(self, context):
        """Convert the context dictionary to a feature vector."""
        raise NotImplementedError("Subclasses must implement preprocess_context")

    def select_arm(self, context):
        """Select an arm based on the context."""
        raise NotImplementedError("Subclasses must implement select_arm")

    def update(self, context, arm, reward):
        """Update the model based on the observed context, arm, and reward."""
        raise NotImplementedError("Subclasses must implement update")

    def add_arm(self):
        """Add a new arm to the algorithm."""
        self.n_arms += 1

    def reset(self):
        """Reset the model to its initial state."""
        self.__init__(self.n_arms, self.context_dim)


class ContextProcessor:
    """Utility class to preprocess context features"""

    def __init__(self):
        """Initialize the context processor with encoders for categorical features"""
        self.currency_encoder = None
        self.time_encoder = None
        self.fitted = False

    def fit(self, contexts):
        """
        Fit the encoders on the given contexts.

        Args:
            contexts: List of context dictionaries
        """
        currency_pairs = [c["currency_pair"] for c in contexts]
        time_of_day = [c["time_of_day"] for c in contexts]

        # Use sparse_output for newer scikit-learn versions
        self.currency_encoder = OneHotEncoder(sparse_output=False)
        self.time_encoder = OneHotEncoder(sparse_output=False)

        self.currency_encoder.fit([[cp] for cp in set(currency_pairs)])
        self.time_encoder.fit([[t] for t in set(time_of_day)])

        self.fitted = True

        # Calculate context dimension
        sample_vector = self.transform(contexts[0])
        return len(sample_vector)

    def transform(self, context):
        """
        Transform a context dictionary into a feature vector.

        Args:
            context: Dictionary with context features

        Returns:
            context_vector: Numpy array of preprocessed features
        """
        if not self.fitted:
            raise ValueError("Context processor must be fitted before transform")

        # Extract features (excluding date which is just an identifier)
        currency_pair = context["currency_pair"]
        volatility = context["volatility"]
        size = context["size"]
        time_of_day = context["time_of_day"]

        # Encode categorical features
        currency_encoded = self.currency_encoder.transform([[currency_pair]])
        time_encoded = self.time_encoder.transform([[time_of_day]])

        # Combine all features
        numeric_features = np.array([volatility, size / 1000])  # Normalize size
        context_vector = np.concatenate(
            [currency_encoded.flatten(), time_encoded.flatten(), numeric_features]
        )

        return context_vector
