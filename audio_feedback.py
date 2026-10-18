from collections import deque
import threading
import wave

import numpy as np


class FootstepAudioDetector:
    """Detect HL1 footstep samples in the Windows speaker loopback stream."""

    SAMPLE_RATE = 11025
    FRAME_SIZE = 256
    FRAME_HOP = 128
    CAPTURE_BLOCK_SIZE = 1024
    BUFFER_SECONDS = 1.0
    REARM_THRESHOLD = 0.62

    def __init__(self, template_dir, threshold=0.82):
        self.template_dir = template_dir
        self.threshold = threshold
        self.templates = self._load_templates()
        self._audio_buffer = deque(
            maxlen=int(self.SAMPLE_RATE * self.BUFFER_SECONDS)
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._event_count = 0
        self._consumed_event_count = 0
        self._last_similarity = 0.0
        self._armed = True
        self._error = None
        self._speaker_name = None

        try:
            import soundcard as sc

            speaker = sc.default_speaker()
            self._speaker_name = speaker.name
            self._loopback = sc.get_microphone(
                id=str(speaker.name), include_loopback=True
            )
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="hl1-footstep-audio",
                daemon=True,
            )
            self._thread.start()
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"

    @property
    def status(self):
        if self._error:
            return f"脚步声检测未启动: {self._error}"
        return f"脚步声检测已启动，监听扬声器回环: {self._speaker_name}"

    def _load_templates(self):
        templates = []
        for number in range(1, 5):
            path = self.template_dir / f"pl_step{number}.wav"
            if not path.is_file():
                raise FileNotFoundError(f"Footstep template not found: {path}")
            with wave.open(str(path), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                frame_count = wav_file.getnframes()
                audio = self._decode_pcm(wav_file.readframes(frame_count), sample_width)

            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            if sample_rate != self.SAMPLE_RATE:
                source_times = np.arange(audio.size, dtype=np.float32) / sample_rate
                target_size = max(1, round(audio.size * self.SAMPLE_RATE / sample_rate))
                target_times = np.arange(target_size, dtype=np.float32) / self.SAMPLE_RATE
                audio = np.interp(target_times, source_times, audio).astype(np.float32)

            features = self._spectral_features(audio)
            norm = float(np.linalg.norm(features))
            if norm > 0:
                templates.append((features / norm).astype(np.float32))
        if len(templates) != 4:
            raise ValueError("Could not build all four footstep templates")
        return templates

    @staticmethod
    def _decode_pcm(raw, sample_width):
        if sample_width == 1:
            return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        if sample_width == 2:
            return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if sample_width == 3:
            data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            values = data[:, 0] | (data[:, 1] << 8) | (data[:, 2] << 16)
            values = (values ^ 0x800000) - 0x800000
            return values.astype(np.float32) / 8388608.0
        if sample_width == 4:
            return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    @classmethod
    def _spectral_features(cls, audio):
        if audio.size < cls.FRAME_SIZE:
            audio = np.pad(audio, (0, cls.FRAME_SIZE - audio.size))
        frames = np.lib.stride_tricks.sliding_window_view(audio, cls.FRAME_SIZE)
        frames = frames[:: cls.FRAME_HOP]
        window = np.hanning(cls.FRAME_SIZE).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
        return np.log1p(spectrum[:, 1:])

    def _capture_loop(self):
        try:
            # WASAPI loopback needs multiple channels; SoundCard documents that
            # single-channel loopback capture can return invalid data on Windows.
            with self._loopback.recorder(
                samplerate=self.SAMPLE_RATE,
                channels=2,
                blocksize=self.CAPTURE_BLOCK_SIZE,
            ) as recorder:
                while not self._stop_event.is_set():
                    samples = recorder.record(numframes=self.CAPTURE_BLOCK_SIZE)
                    if samples.ndim == 2:
                        samples = samples.mean(axis=1)
                    self._process_audio(samples.astype(np.float32, copy=False))
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            print(f"脚步声回环捕获失败: {self._error}")

    def _process_audio(self, samples):
        with self._lock:
            self._audio_buffer.extend(samples.tolist())
            audio = np.asarray(self._audio_buffer, dtype=np.float32)
            if audio.size < self.SAMPLE_RATE // 3:
                return
            audio_features = self._spectral_features(audio)
            best_similarity = 0.0
            for template in self.templates:
                frame_count = template.shape[0]
                if audio_features.shape[0] < frame_count:
                    continue
                for start in range(audio_features.shape[0] - frame_count + 1):
                    candidate = audio_features[start : start + frame_count]
                    candidate_norm = float(np.linalg.norm(candidate))
                    if candidate_norm == 0:
                        continue
                    similarity = float(np.sum(candidate * template) / candidate_norm)
                    best_similarity = max(best_similarity, similarity)

            self._last_similarity = best_similarity
            if best_similarity < self.REARM_THRESHOLD:
                self._armed = True
            elif best_similarity >= self.threshold and self._armed:
                self._event_count += 1
                self._armed = False

    def consume_new_event(self):
        with self._lock:
            detected = self._event_count > self._consumed_event_count
            self._consumed_event_count = self._event_count
            return detected, self._last_similarity

    def discard_pending_events(self):
        with self._lock:
            self._consumed_event_count = self._event_count

    def close(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
