# PPG-LabBench: third-party notices

The code of this repository is under the [MIT licence](LICENSE). It bundles no third-party code and no model weights. Two modules re-implement published network architectures so that their weights load, and the
notices for those are below. You download the weights yourself, under the terms in [DATA.md](DATA.md).

## PaPaGei

`src/ppglabbench/features/papagei.py` re-implements the ResNet1D backbone of [PaPaGei](https://github.com/Nokia-Bell-Labs/papagei-foundation-model), from its `models/resnet.py`, and follows its preprocessing. That
repository states its licence in two ways:

- Its `LICENSE` file: BSD 3-Clause, reproduced below.
- The header of `models/resnet.py`: `© 2024 Nokia`, `Licensed under the BSD 3 Clause Clear License`, `SPDX-License-Identifier: BSD-3-Clause-Clear`. That licence has the same conditions and adds that it grants no
  patent rights.

```text
BSD 3-Clause License

Copyright (c) 2024, Device Research at Nokia Bell Labs

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Pulse-PPG

`src/ppglabbench/features/pulseppg.py` re-implements the ResNet1D of [Pulse-PPG](https://github.com/maxxu05/pulseppg), from its `pulseppg/nets/ResNet1D/ResNet1D_Net.py`.

```text
MIT License

Copyright (c) 2025 maxxu05

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## resnet1d

The networks of both repositories above come from [resnet1d](https://github.com/hsd1503/resnet1d) by Shenda Hong, under the [Apache License 2.0](LICENSES/Apache-2.0.txt). The two modules above are modified versions:
rewritten for inference only, without dropout, grouped convolutions or training code, and with the module names kept so that the weights load.

## pyPPG

[pyPPG](https://github.com/godamartonaron/GODA_pyPPG) is under GPL-3.0. It is not imported and none of its code is copied. PaPaGei's preprocessing uses pyPPG's default filter parameters, and the two modules above use
the same values through SciPy: a Chebyshev-II band-pass of 0.5-12 Hz, order 4, and a 50 ms moving average.

## Dependencies

Installed by you, not bundled, each under its own licence: the packages in `pyproject.toml`, and PyTorch, which has a BSD-style licence (the `LICENSE` file in its [repository](https://github.com/pytorch/pytorch) and its
packages). PyTorch is needed by the PaPaGei and Pulse-PPG embeddings and by the small CNN in `ppglabbench.features.cnn`. The CNN's network is written for this project, and its weights are not distributed.
