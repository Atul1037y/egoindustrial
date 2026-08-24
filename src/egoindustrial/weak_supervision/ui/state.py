"""Session state management for Streamlit UI."""

from typing import Any

import pandas as pd
import streamlit as st


class SessionState:
    """Singleton session state manager."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.labels_df: pd.DataFrame | None = None
        self.verifications: dict[int, bool] = {}
        self.corrections: dict[int, dict[str, Any]] = {}
        self.verb_to_idx: dict[str, int] = {}
        self.noun_to_idx: dict[str, int] = {}

    @classmethod
    def get(cls):
        if "session_state" not in st.session_state:
            st.session_state.session_state = cls()
        return st.session_state.session_state

    def load_labels(self, uploaded_file) -> None:
        """Load pseudo-labels from uploaded CSV."""
        self.labels_df = pd.read_csv(uploaded_file)

        # Build class mappings
        if "verb_class" in self.labels_df.columns:
            verbs = sorted(self.labels_df["verb_class"].unique())
            self.verb_to_idx = {v: i for i, v in enumerate(verbs)}

        if "noun_class" in self.labels_df.columns:
            nouns = sorted(self.labels_df["noun_class"].unique())
            self.noun_to_idx = {n: i for i, n in enumerate(nouns)}

        # Reset verifications
        self.verifications = {}
        self.corrections = {}

    def apply_filters(
        self,
        min_conf: float,
        verb_filter: str,
        noun_filter: str,
    ) -> pd.DataFrame:
        """Apply filters to labels."""
        if self.labels_df is None:
            return pd.DataFrame()

        df = self.labels_df.copy()

        # Confidence filter
        df = df[(df["verb_confidence"] >= min_conf) & (df["noun_confidence"] >= min_conf)]

        # Class filters
        if verb_filter != "All":
            df = df[df["verb_class"] == verb_filter]
        if noun_filter != "All":
            df = df[df["noun_class"] == noun_filter]

        return df

    def get_filtered(self) -> pd.DataFrame:
        """Get currently filtered dataframe."""
        # This would need to be called after apply_filters
        # For simplicity, return all if no filters applied
        return self.labels_df if self.labels_df is not None else pd.DataFrame()

    def is_verified(self, idx: int) -> bool:
        return idx in self.verifications

    def get_verification(self, idx: int) -> bool | None:
        return self.verifications.get(idx)

    def mark_verified(self, idx: int, accepted: bool) -> None:
        self.verifications[idx] = accepted

    def apply_correction(self, idx: int, corrections: dict[str, Any]) -> None:
        self.corrections[idx] = corrections
        # Also update the dataframe
        if self.labels_df is not None:
            for key, value in corrections.items():
                self.labels_df.at[idx, key] = value

    @property
    def verified_count(self) -> int:
        return len(self.verifications)

    @property
    def accepted_count(self) -> int:
        return sum(1 for v in self.verifications.values() if v)

    @property
    def rejected_count(self) -> int:
        return sum(1 for v in self.verifications.values() if not v)

    def get_verified_dataframe(self) -> pd.DataFrame:
        """Get dataframe with only verified (accepted) labels."""
        if self.labels_df is None:
            return pd.DataFrame()

        verified_indices = [i for i, v in self.verifications.items() if v]
        if not verified_indices:
            return pd.DataFrame()

        df = self.labels_df.loc[verified_indices].copy()

        # Apply corrections
        for idx, corrections in self.corrections.items():
            if idx in df.index:
                for key, value in corrections.items():
                    df.at[idx, key] = value

        return df
