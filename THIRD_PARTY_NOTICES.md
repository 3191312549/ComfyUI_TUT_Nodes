# Third-party notices

## SesquiLSR

`TUT_SesquiLatentUpscale` includes adapted SesquiLSR inference and latent-format
adapter code from the following fixed upstream revision:

Project: https://github.com/LoganBooker/SesquiLSR

Revision: `befae004248c403f38b76b9f65fd43b901ea3eaa`

Source: https://github.com/LoganBooker/SesquiLSR/tree/befae004248c403f38b76b9f65fd43b901ea3eaa

The node can download the upstream `upscaler_SDXL.safetensors`,
`upscaler_Flux.safetensors`, `upscaler_Flux2.safetensors`, or
`upscaler_Wan21.safetensors` weight on first execution for the selected model
format. These weights are not bundled with this plugin. Ideogram 4 uses the
Flux2 weight.

MIT License

Copyright (c) 2026 SesquiLSR Contributors

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
