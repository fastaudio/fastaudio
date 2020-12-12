# -*- coding: utf-8 -*-
from pkg_resources import DistributionNotFound, get_distribution

import os
import torchaudio

torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
# soundfile is torchaudio backend for windows, all other os use sox_io
backend = "soundfile" if os.name == "nt" else "sox_io"
torchaudio.set_audio_backend(backend)
try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
