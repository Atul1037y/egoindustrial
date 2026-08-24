"""UI components for pseudo-label verification."""

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st


def video_player(row: pd.Series) -> None:
    """Display video player with segment highlighting."""
    video_path = row["video_path"]

    if not Path(video_path).exists():
        st.warning(f"Video not found: {video_path}")
        return

    # Extract segment for preview
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp.name, fourcc, fps, (width, height))

            start = int(row["start_frame"])
            end = int(row["end_frame"])

            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(end - start + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                # Add overlay
                cv2.putText(
                    frame,
                    f"{row['verb_class']} + {row['noun_class']}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Conf: {row['verb_confidence']:.2f} / {row['noun_confidence']:.2f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                out.write(frame)

            cap.release()
            out.release()

            st.video(tmp.name)

        except Exception as e:
            st.error(f"Error playing video: {e}")
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def label_editor(state, idx: int, row: pd.Series) -> None:
    """Edit/verify pseudo-label."""
    st.subheader("✅ Verification")

    # Current prediction
    st.write(f"**Predicted:** {row['verb_class']} + {row['noun_class']}")
    st.write(
        f"**Confidence:** Verb: {row['verb_confidence']:.2f} | Noun: {row['noun_confidence']:.2f}"
    )

    # Verification status
    verified = state.is_verified(idx)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Accept", key=f"accept_{idx}", disabled=verified, type="primary"):
            state.mark_verified(idx, True)
            st.rerun()

    with col2:
        if st.button("❌ Reject", key=f"reject_{idx}", disabled=verified):
            state.mark_verified(idx, False)
            st.rerun()

    if verified:
        if state.get_verification(idx):
            st.success("✅ **Accepted**")
        else:
            st.error("❌ **Rejected**")

    # Manual correction
    with st.expander("✏️ Manual Correction"):
        verb_options = ["keep"] + sorted(state.labels_df["verb_class"].unique().tolist())
        noun_options = ["keep"] + sorted(state.labels_df["noun_class"].unique().tolist())

        new_verb = st.selectbox("Verb", verb_options, key=f"verb_{idx}")
        new_noun = st.selectbox("Noun", noun_options, key=f"noun_{idx}")

        if st.button("Apply Correction", key=f"correct_{idx}"):
            corrections = {}
            if new_verb != "keep":
                corrections["verb_class"] = new_verb
                corrections["verb_label"] = state.verb_to_idx[new_verb]
            if new_noun != "keep":
                corrections["noun_class"] = new_noun
                corrections["noun_label"] = state.noun_to_idx[new_noun]

            state.apply_correction(idx, corrections)
            st.success("Correction applied!")
            st.rerun()


def stats_panel(state) -> None:
    """Display verification statistics."""
    total = len(state.labels_df)
    verified = state.verified_count
    accepted = state.accepted_count
    rejected = state.rejected_count
    pending = total - verified

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total", total)
    col2.metric("Verified", verified)
    col3.metric("✅ Accepted", accepted)
    col4.metric("❌ Rejected", rejected)
    col5.metric("⏳ Pending", pending)

    if verified > 0:
        acceptance_rate = accepted / verified * 100
        st.progress(verified / total)
        st.caption(f"Acceptance rate: {acceptance_rate:.1f}%")
