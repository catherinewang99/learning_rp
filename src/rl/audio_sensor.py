"""Audio sensor: a person talking at the goal, heard from the agent's position.

The goal loops a speech clip. What the agent receives at distance d is
    y = clip_segment * gain(d) + noise(d),   gain = ref/(ref + d),
    noise sigma = noise_base + noise_slope * d
— louder AND cleaner near the goal, softer and noisier far away.

One observation per env step: the log-mel spectrogram (C, n_mels, F) of the
audio received over the LAST `window_steps` steps of the trajectory.
Binaural (default, C=2): head-shadow ILD + a pinna-like front bias make the
left/right level difference an INSTANTANEOUS direction cue (ITD is omitted —
phase does not survive log-mel). Mono ablation (C=1): the loudness trend
across the window is the only directional signal.

Rendering is a pure function of ((distance, bearing) history, clip phase,
noise seed), so cross-rendering the teacher's trajectory and building
deterministic probe banks need only pose histories — no simulator state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class AudioConfig:
    clip_path: str | None = None      # wav of a person talking; None -> torchaudio asset
    sample_rate: int = 16000
    step_seconds: float = 0.1         # audio arriving per env step
    window_steps: int = 8             # observation spans this many past steps
    n_mels: int = 64
    n_fft: int = 512
    hop: int = 256
    ref_distance: float = 0.75        # gain = ref / (ref + d)
    noise_base: float = 0.005
    noise_slope: float = 0.02         # sigma grows with distance
    binaural: bool = True             # 2-channel (L, R) via head-shadow ILD;
    #   False = the mono loudness-trend-only ablation
    shadow_gain: float = 0.4          # ILD strength: ear gains scale by
    #   (1 + front*cos(beta) +- shadow*sin(beta)), beta = source bearing
    front_gain: float = 0.2           # pinna-like front bias (breaks the
    #   front/back ambiguity pure ILD has). ITD deliberately omitted: ~9
    #   samples at 0.2 m spacing is invisible after log-mel (phase discarded).
    log_offset: float = 1e-6
    norm_center: float = -8.0         # fixed affine so typical log-mel ~ [-1, 1];
    norm_scale: float = 4.0           #   NOT per-sample standardized — absolute
    #   loudness IS the signal.


DEFAULT_ASSET = "tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"


class AudioSensor:
    def __init__(self, cfg: AudioConfig, seed: int = 0, clip: np.ndarray | None = None):
        """``clip``: inject a waveform directly (tests / synthetic sources);
        None loads cfg.clip_path or the default torchaudio speech asset."""
        self.cfg = cfg
        self.step_samples = int(cfg.sample_rate * cfg.step_seconds)
        self.rng = np.random.default_rng(seed)
        self.clip = (clip / (np.abs(clip).max() + 1e-8)) if clip is not None else self._load_clip()
        self._mel = None  # lazy torchaudio transform

    def _load_clip(self) -> np.ndarray:
        import torchaudio

        path = self.cfg.clip_path
        if path is None:
            path = torchaudio.utils.download_asset(DEFAULT_ASSET)
        wav, sr = torchaudio.load(str(Path(path)))
        wav = wav.mean(dim=0)
        if sr != self.cfg.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.cfg.sample_rate)(wav)
        clip = wav.numpy()
        assert len(clip) >= self.step_samples, "speech clip shorter than one env step"
        return clip / (np.abs(clip).max() + 1e-8)

    @property
    def channels(self) -> int:
        return 2 if self.cfg.binaural else 1

    def gain(self, d: np.ndarray | float) -> np.ndarray | float:
        return self.cfg.ref_distance / (self.cfg.ref_distance + d)

    def ear_gains(self, d: float, beta: float) -> np.ndarray:
        """Per-channel amplitude at distance d, source bearing beta (radians,
        0 = ahead, +pi/2 = left). Mono: omnidirectional."""
        base = float(self.gain(d))
        if not self.cfg.binaural:
            return np.array([base])
        front = self.cfg.front_gain * np.cos(beta)
        shadow = self.cfg.shadow_gain * np.sin(beta)
        return base * np.clip([1 + front + shadow, 1 + front - shadow], 0.05, None)

    def noise_sigma(self, d: np.ndarray | float) -> np.ndarray | float:
        return self.cfg.noise_base + self.cfg.noise_slope * d

    # -- waveform & spectrogram ------------------------------------------------

    def received_window(self, cues: np.ndarray, phase: int,
                        rng: np.random.Generator | None = None) -> np.ndarray:
        """(C, W*step_samples) waveform for the last W steps. cues (W, 2) =
        [distance, bearing] at each step (oldest first; bearing was measured
        with that step's heading); phase = clip sample index at the window
        START. Deterministic given (cues, phase, rng)."""
        rng = self.rng if rng is None else rng
        s, clip = self.step_samples, self.clip
        chans = [[] for _ in range(self.channels)]
        for j, (d, beta) in enumerate(np.atleast_2d(cues)):
            start = (phase + j * s) % len(clip)
            seg = np.take(clip, np.arange(start, start + s), mode="wrap")
            for c, g in enumerate(self.ear_gains(float(d), float(beta))):
                chans[c].append(seg * g + rng.normal(0.0, self.noise_sigma(d), s))
        return np.stack([np.concatenate(ch) for ch in chans])

    def spectrogram(self, waveform: np.ndarray) -> torch.Tensor:
        """(C, n_mels, F) log-mel with FIXED affine normalization.
        waveform: (C, samples)."""
        import torchaudio

        if self._mel is None:
            self._mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.cfg.sample_rate, n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop, n_mels=self.cfg.n_mels,
            )
        mel = self._mel(torch.from_numpy(waveform).float())
        logmel = torch.log(mel + self.cfg.log_offset)
        return (logmel - self.cfg.norm_center) / self.cfg.norm_scale

    @staticmethod
    def _as_cues(history: np.ndarray) -> np.ndarray:
        """Accept (W,) distances (bearing 0) or (W, 2) [d, beta] rows."""
        h = np.asarray(history, dtype=np.float64)
        if h.ndim == 1:
            h = np.stack([h, np.zeros_like(h)], axis=1)
        return h

    def observe_traj(self, cue_history: np.ndarray, end_phase: int,
                     rng: np.random.Generator | None = None) -> torch.Tensor:
        """Observation at the END of a cue history (last W rows used; padded by
        repeating the oldest if shorter — e.g. right after reset).
        end_phase = clip sample index at the END of the window."""
        w = self.cfg.window_steps
        cues = self._as_cues(cue_history)
        if len(cues) < w:
            cues = np.concatenate([np.repeat(cues[:1], w - len(cues), axis=0), cues])
        cues = cues[-w:]
        start_phase = (end_phase - w * self.step_samples) % len(self.clip)
        return self.spectrogram(self.received_window(cues, start_phase, rng))

    def observe_stationary(self, dist: float, bearing: float = 0.0, phase: int = 0,
                           rng: np.random.Generator | None = None) -> torch.Tensor:
        """Probe-bank rendering: as if the agent sat at this (distance,
        bearing) for the whole window (documented statistics mismatch vs.
        moving rollouts)."""
        cues = np.tile([dist, bearing], (self.cfg.window_steps, 1))
        return self.observe_traj(cues, phase, rng)

    @property
    def frames(self) -> int:
        """Spectrogram time-axis length for the configured window."""
        return self.cfg.window_steps * self.step_samples // self.cfg.hop + 1
