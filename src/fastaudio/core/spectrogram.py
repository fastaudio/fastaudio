import warnings
from dataclasses import asdict, is_dataclass
from inspect import signature

import torchaudio
from fastai.data.core import TensorImageBase
from fastai.imports import inspect, partial, plt
from fastcore.transform import Pipeline, Transform
from fastcore.utils import L, add_props, delegates, ifnone
from librosa.display import specshow

from .signal import AudioTensor

# from fastai.data.all import *
# from .signal import *


class AudioSpectrogram(TensorImageBase):
    @classmethod
    def create(cls, sg_tensor, settings=None):
        audio_sg = cls(sg_tensor)
        audio_sg._settings = settings
        return audio_sg

    @property
    def duration(self):
        # spectrograms round up length to fill incomplete columns,
        # so we subtract 0.5 to compensate, wont be exact
        return (self.hop_length * (self.shape[-1] - 0.5)) / self.sr

    height, width = add_props(lambda i, self: self.shape[i + 1], n=2)
    # using the line below instead of above will fix show_batch but break
    # multichannel/delta display
    # nchannels, height, width = add_props(lambda i, self: self.shape[i], n=3)

    def __getattr__(self, name):
        if name == "settings":
            return self._settings
        if not name.startswith("_"):
            return self._settings[name]
        raise AttributeError(
            f"{self.__class__.__name__} object has no attribute {name}"
        )

    def show(self, ctx=None, ax=None, figsize=None, **kwargs):
        show_spectrogram(self, ctx=ctx, ax=ax, figsize=figsize, **kwargs)
        plt.show()


def show_spectrogram(sg, ax, ctx, figsize, **kwargs):
    ax = ifnone(ax, ctx)
    nchannels = sg.nchannels
    r, c = nchannels, sg.data.shape[0] // nchannels
    proper_kwargs = get_usable_kwargs(
        specshow, sg._settings, exclude=["ax", "kwargs", "data"]
    )
    if r == 1 and c == 1:
        _show_spectrogram(sg, ax, proper_kwargs, **kwargs)
        plt.title("Channel 0 Image 0: {} X {}px".format(*sg.shape[-2:]))
    else:
        if figsize is None:
            figsize = (4 * c, 3 * r)
        if ax is None:
            _, ax = plt.subplots(r, c, figsize=figsize)
        for i, channel in enumerate(sg.data):
            if r == 1:
                cur_ax = ax[i % c]
            elif c == 1:
                cur_ax = ax[i % r]
            else:
                cur_ax = ax[i // c, i % c]
            width, height = sg.shape[-2:]
            cur_ax.set_title(f"Channel {i//c} Image {i%c}: {width} X {height}px")
            specshow(channel.numpy(), ax=cur_ax, **sg._show_args, **proper_kwargs)
            # plt.colorbar(z, ax=cur_ax)
            # ax=plt.gca() #get the current axes
            # get the mappable, the 1st and the 2nd are the x and y axes
            # PCM=ax.get_children()[2]
            # plt.colorbar(PCM, ax=ax, format='%+2.0f dB')


def _show_spectrogram(sg, ax, proper_kwargs, **kwargs):
    if "mel" not in sg._settings:
        y_axis = None
    else:
        y_axis = "mel" if sg.mel else "linear"
    proper_kwargs.update({"x_axis": "time", "y_axis": y_axis})
    _ = specshow(sg.data.squeeze(0).numpy(), **sg._show_args, **proper_kwargs)
    fmt = "%+2.0f dB" if "to_db" in sg._settings and sg.to_db else "%+2.0f"
    plt.colorbar(format=fmt)


_GenSpec = torchaudio.transforms.Spectrogram
_GenMelSpec = torchaudio.transforms.MelSpectrogram
_GenMFCC = torchaudio.transforms.MFCC
_ToDB = torchaudio.transforms.AmplitudeToDB


class AudioToSpec(Transform):
    def __init__(self, pipe, settings):
        self.pipe = pipe
        self.settings = settings

    @classmethod
    def from_cfg(cls, audio_cfg):
        cfg = asdict(audio_cfg) if is_dataclass(audio_cfg) else dict(audio_cfg)
        transformer = SpectrogramTransformer(mel=cfg.pop("mel"), to_db=cfg.pop("to_db"))
        return transformer(**cfg)

    def encodes(self, audio: AudioTensor):
        self.settings.update({"sr": audio.sr, "nchannels": audio.nchannels})
        return AudioSpectrogram.create(
            self.pipe(audio.data), settings=dict(self.settings)
        )


def SpectrogramTransformer(mel=True, to_db=True):
    sg_type = {"mel": mel, "to_db": to_db}
    transforms = _get_transform_list(sg_type)
    pipe_noargs = partial(fill_pipeline, sg_type=sg_type, transform_list=transforms)
    pipe_noargs.__signature__ = _get_signature(transforms)
    return pipe_noargs


def _get_transform_list(sg_type):
    """Builds a list of higher-order transforms with no arguments"""
    transforms = L()
    if sg_type["mel"]:
        transforms += _GenMelSpec
    else:
        transforms += _GenSpec
    if sg_type["to_db"]:
        transforms += _ToDB
    return transforms


def fill_pipeline(transform_list, sg_type, **kwargs):
    """Adds correct args to each transform"""
    kwargs = _override_bad_defaults(dict(kwargs))
    function_list = L()
    settings = {}
    for f in transform_list:
        usable_kwargs = get_usable_kwargs(f, kwargs)
        function_list += f(**usable_kwargs)
        settings.update(usable_kwargs)
    warn_unused(kwargs, settings)
    return AudioToSpec(Pipeline(function_list), settings={**sg_type, **settings})


def _get_signature(transforms):
    """Looks at transform list and extracts all valid args for tab completion"""
    delegations = [delegates(to=f, keep=True) for f in transforms]
    out = lambda **kwargs: None  # noqa: E731
    for d in delegations:
        out = d(out)
    return signature(out)


def _override_bad_defaults(kwargs):
    if "n_fft" not in kwargs or kwargs["n_fft"] is None:
        kwargs["n_fft"] = 1024
    if "win_length" not in kwargs or kwargs["win_length"] is None:
        kwargs["win_length"] = kwargs["n_fft"]
    if "hop_length" not in kwargs or kwargs["hop_length"] is None:
        kwargs["hop_length"] = int(kwargs["win_length"] / 2)
    return kwargs


def warn_unused(all_kwargs, used_kwargs):
    unused_kwargs = set(all_kwargs.keys()) - set(used_kwargs.keys())
    for kwarg in unused_kwargs:
        warnings.warn(f"{kwarg} is not a valid arg name and was not used")


def get_usable_kwargs(func, kwargs, exclude=None):
    exclude = ifnone(exclude, [])
    defaults = {
        k: v.default
        for k, v in inspect.signature(func).parameters.items()
        if k not in exclude
    }
    usable = {k: v for k, v in kwargs.items() if k in defaults}
    return {**defaults, **usable}


@delegates(_GenMFCC.__init__)
class AudioToMFCC(Transform):
    def __init__(self, **kwargs):
        func_args = get_usable_kwargs(_GenMFCC, kwargs, [])
        self.transformer = _GenMFCC(**func_args)
        self.settings = func_args

    @classmethod
    def from_cfg(cls, audio_cfg):
        cfg = asdict(audio_cfg) if is_dataclass(audio_cfg) else audio_cfg
        return cls(**cfg)

    def encodes(self, x: AudioTensor):
        sg_settings = {"sr": x.sr, "nchannels": x.nchannels, **self.settings}
        return AudioSpectrogram.create(
            self.transformer(x).detach(), settings=sg_settings
        )
