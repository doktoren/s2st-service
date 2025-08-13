# TODO

## Current behavior
- The WebSocket handler reads one message at a time and appends decoded audio frames to an in-memory buffer.
- Voice activity detection checks for 400 ms of trailing silence; once detected the buffered audio is sent for translation.
- Translation runs to completion and the server emits a single `audio_chunk` followed by `end_of_audio` for each utterance.
- If the client sends a `close` message any buffered audio is dropped and no final translation is produced.

## Intended behavior
- When the test page is started the browser should stream 20 ms audio chunks over a WebSocket connection.
- The backend performs VAD and, after a pause, sends the non-empty audio so far to the model.
- As soon as the model begins producing audio—even partially—the backend streams the result to the browser.  Chunks may be transmitted faster than real time; the browser should play at normal speed until finished.
- While a translation is running the server should not start another one; it should pause websocket reads or otherwise ensure only one translation runs at a time.
- Clicking stop on the test page should send roughly one second of silence to trigger the final VAD segment.

## Changes needed
- Stream model output incrementally instead of buffering the entire translated utterance.  Consider chunking the generated waveform and emitting each piece as soon as available.
- Avoid dropping residual audio on client close; emit any buffered speech before closing.
- Introduce explicit back-pressure or a queue so that incoming audio is not processed while a translation is ongoing.
- Ensure a final silent segment is appended when the client stops recording.

## Simplifications
- If client-side chunking is unnecessary, drop the `client_chunked` path and the related negotiation logic to reduce complexity.
- Supporting only one codec and sample rate would eliminate a substantial amount of conversion code and resampling.
- The strategy parser prints and suppresses exceptions; replacing this with straightforward validation would clarify the setup path.
