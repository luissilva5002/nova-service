# WebSocket streaming protocol

Clients may add `"stream": true` to a text message. Chat turns send zero or
more `text_delta` frames, zero or more `audio_chunk` frames (`seq`,
`sample_rate`, and base64 PCM16 payload), the normal final `text` and
`metrics` frames, and then `audio_done`. Clients omitting `stream` continue
to receive the existing `text`, `metrics`, optional `open_url`, and `audio`
frames.
