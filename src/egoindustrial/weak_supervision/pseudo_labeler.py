"""Pseudo-labeling engine for unlabeled video frames."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from egoindustrial.data.transforms import get_transforms
from egoindustrial.inference.tensorrt_engine import TensorRTEngine


class PseudoLabeler:
    """Generate pseudo-labels for unlabeled videos using trained model."""

    def __init__(
        self,
        engine: TensorRTEngine,
        transforms,
        verb_classes: list[str],
        noun_classes: list[str],
        confidence_threshold: float = 0.9,
        entropy_threshold: float = 0.5,
        min_frames: int = 16,
        stride: int = 8,
    ):
        self.engine = engine
        self.transforms = transforms
        self.verb_classes = verb_classes
        self.noun_classes = noun_classes
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.min_frames = min_frames
        self.stride = stride

    def _compute_entropy(self, probs: np.ndarray) -> float:
        """Compute entropy of probability distribution."""
        probs = probs + 1e-10
        return -np.sum(probs * np.log(probs))

    def _passes_filters(self, verb_probs: np.ndarray, noun_probs: np.ndarray) -> bool:
        """Check if predictions pass confidence/entropy thresholds."""
        verb_conf = verb_probs.max()
        noun_conf = noun_probs.max()
        verb_ent = self._compute_entropy(verb_probs)
        noun_ent = self._compute_entropy(noun_probs)

        return (
            verb_conf >= self.confidence_threshold
            and noun_conf >= self.confidence_threshold
            and verb_ent <= self.entropy_threshold
            and noun_ent <= self.entropy_threshold
        )

    def label_video(
        self,
        video_path: str,
        output_dir: str,
    ) -> list[dict[str, Any]]:
        """Generate pseudo-labels for a single video."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pseudo_labels = []
        frame_buffer = []
        frame_idx = 0

        with tqdm(total=total_frames, desc=f"Labeling {Path(video_path).name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_buffer.append(frame_rgb)

                # Process when we have enough frames
                if len(frame_buffer) >= self.min_frames:
                    if len(frame_buffer) == self.min_frames or frame_idx % self.stride == 0:
                        # Convert to tensor
                        clip = np.stack(frame_buffer[-self.min_frames :])
                        clip_tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0
                        clip_tensor = self.transforms(clip_tensor)

                        # Inference
                        outputs = self.engine.infer(clip_tensor.numpy()[None])

                        verb_probs = torch.softmax(torch.from_numpy(outputs[0][0]), dim=-1).numpy()
                        noun_probs = torch.softmax(torch.from_numpy(outputs[1][0]), dim=-1).numpy()
                        action_probs = torch.softmax(
                            torch.from_numpy(outputs[2][0]), dim=-1
                        ).numpy()

                        if self._passes_filters(verb_probs, noun_probs):
                            verb_idx = verb_probs.argmax()
                            noun_idx = noun_probs.argmax()
                            action_idx = action_probs.argmax()

                            pseudo_labels.append(
                                {
                                    "video_path": video_path,
                                    "start_frame": frame_idx - self.min_frames + 1,
                                    "end_frame": frame_idx,
                                    "start_time": (frame_idx - self.min_frames + 1) / fps,
                                    "end_time": frame_idx / fps,
                                    "verb_label": int(verb_idx),
                                    "verb_class": self.verb_classes[verb_idx]
                                    if verb_idx < len(self.verb_classes)
                                    else str(verb_idx),
                                    "verb_confidence": float(verb_probs[verb_idx]),
                                    "noun_label": int(noun_idx),
                                    "noun_class": self.noun_classes[noun_idx]
                                    if noun_idx < len(self.noun_classes)
                                    else str(noun_idx),
                                    "noun_confidence": float(noun_probs[noun_idx]),
                                    "action_label": int(action_idx),
                                    "action_confidence": float(action_probs[action_idx]),
                                }
                            )

                    # Slide window
                    if len(frame_buffer) > self.min_frames:
                        frame_buffer = frame_buffer[-(self.min_frames - 1) :]

                frame_idx += 1
                pbar.update(1)

        cap.release()

        # Save
        output_path = Path(output_dir) / f"{Path(video_path).stem}_pseudo.csv"
        import pandas as pd

        df = pd.DataFrame(pseudo_labels)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(pseudo_labels)} pseudo-labels to {output_path}")

        return pseudo_labels

    def label_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".MP4"),
    ) -> list[dict[str, Any]]:
        """Label all videos in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        video_files = [f for f in input_path.iterdir() if f.suffix in extensions]

        all_labels = []
        for video_file in video_files:
            try:
                labels = self.label_video(str(video_file), str(output_path))
                all_labels.extend(labels)
            except Exception as e:
                print(f"Error processing {video_file}: {e}")

        # Combined CSV
        import pandas as pd

        combined_df = pd.DataFrame(all_labels)
        combined_path = output_path / "all_pseudo_labels.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"Total pseudo-labels: {len(all_labels)}")

        return all_labels


def create_pseudo_labeler_from_checkpoint(
    checkpoint_path: str,
    engine_path: str,
    verb_classes: list[str],
    noun_classes: list[str],
    **kwargs,
) -> PseudoLabeler:
    """Factory to create PseudoLabeler from training checkpoint."""
    # Export to ONNX -> TensorRT if needed
    if not Path(engine_path).exists():
        from egoindustrial.inference.export_onnx import export_to_onnx
        from egoindustrial.inference.tensorrt_engine import build_tensorrt_engine

        onnx_path = engine_path.replace(".engine", ".onnx")
        export_to_onnx(checkpoint_path, onnx_path)
        build_tensorrt_engine(onnx_path, engine_path, precision="fp16")

    engine = TensorRTEngine(engine_path)
    transforms = get_transforms({"is_train": False, "model_type": "videomae"})

    return PseudoLabeler(engine, transforms, verb_classes, noun_classes, **kwargs)
