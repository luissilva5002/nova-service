# WebSocket streaming protocol

Clients may add `"stream": true` to a text message. Chat turns send zero or
more `text_delta` frames, zero or more `audio_chunk` frames (`seq`,
`sample_rate`, and base64 PCM16 payload), the normal final `text` and
`metrics` frames, and then `audio_done`. Clients omitting `stream` continue
to receive the existing `text`, `metrics`, optional `open_url`, and `audio`
frames.

If a turn fails on the server, the connection stays open and the client
receives a single `{"type": "error", "text": "..."}` frame instead of the
frames above. Clients should show `text`, discard any partially streamed
message, and be ready to send the next turn. Clients that do not know the
`error` type can ignore it.
