import torchaudio.transforms as transforms
from dataclasses import make_dataclass
from fastai.data.block import TransformBlock
from fastai.data.transforms import IntToFloatTensor
from fastai.imports import Path, partial
from fastcore.meta import delegates
from fastcore.transform import Pipeline
from fastcore.utils import ifnone
from inspect import signature
from torchaudio import save as save_audio

from ..augment.preprocess import Resample
from ..augment.signal import DownmixMono, ResizeSignal
from .signal import AudioTensor, get_audio_files


def audio_item_tfms(sample_rate=16000, force_mono=True, crop_signal_to=None):
    """
    Basic audio preprocessing transforms.
    """
    tfms = []
    if sample_rate is not None:
        tfms.append(Resample(sample_rate))
    if force_mono:
        tfms.append(DownmixMono())
    if crop_signal_to is not None:
        tfms.append(ResizeSignal(duration=crop_signal_to))
    return tfms


class PreprocessAudio:
    """
    Creates an audio tensor and run the basic preprocessing transforms on it.
    Used while preprocessing the audios, this is not a `Transform`.
    """

    @delegates(audio_item_tfms)
    def __init__(self, **kwargs):
        self.tfms = Pipeline(audio_item_tfms(**kwargs))

    def __call__(self, x):
        audio = AudioTensor.create(x)
        return self.tfms(audio)


@delegates(PreprocessAudio, keep=True)
def preprocess_audio_folder(path, folders=None, output_dir=None, **kwargs):
    "Preprocess audio files in `path` in parallel using `n_workers`"
    path = Path(path)
    fnames = get_audio_files(path, recurse=True, folders=folders)
    output_dir = Path(ifnone(output_dir, path.parent / f"{path.name}_cached"))
    output_dir.mkdir(exist_ok=True)

    pp = PreprocessAudio(**kwargs)

    for i, fil in enumerate(fnames):
        out = output_dir / fnames[i].relative_to(path)
        aud = pp(fil)
        save_audio(str(out), aud, aud.sr)
    return output_dir


class AudioBlock(TransformBlock):
    "A `TransformBlock` for audios"

    @delegates(audio_item_tfms)
    def __init__(self, cache_folder=None, **kwargs):
        item_tfms = audio_item_tfms(**kwargs)
        type_tfm = partial(AudioTensor.create, cache_folder=cache_folder)
        return super().__init__(
            type_tfms=type_tfm, item_tfms=item_tfms, batch_tfms=IntToFloatTensor
        )

    @classmethod
    @delegates(audio_item_tfms, keep=True)
    def from_folder(cls, path, **kwargs):
        "Build a `AudioBlock` from a `path` and caches some intermediary results"
        cache_folder = preprocess_audio_folder(path, **kwargs)
        return cls(cache_folder, **kwargs)


def config_from_func(func, name, **kwargs):
    params = signature(func).parameters.items()
    namespace = {k: v.default for k, v in params}
    namespace.update(kwargs)
    func_config = make_dataclass(name, namespace.keys(), namespace=namespace)
    func_config.__doc__ = ""
    return func_config


class AudioConfig:
    """
    Collection of configurations to build `AudioToSpec` transforms.
    """

    # default configurations from the wrapped function
    # make sure to pass in mel=False as kwarg for non-mel spec
    # and to_db=False for non db spec
    BasicSpectrogram = config_from_func(
        transforms.Spectrogram, "BasicSpectrogram", mel=False, to_db=True
    )
    BasicMelSpectrogram = config_from_func(
        transforms.MelSpectrogram, "BasicMelSpectrogram", mel=True, to_db=True
    )
    BasicMFCC = config_from_func(transforms.MFCC, "BasicMFCC ")
    # special configs with domain-specific defaults

    Voice = config_from_func(
        transforms.MelSpectrogram,
        "Voice",
        mel="True",
        to_db="False",
        f_min=50.0,
        f_max=8000.0,
        n_fft=1024,
        n_mels=128,
        hop_length=128,
    )
