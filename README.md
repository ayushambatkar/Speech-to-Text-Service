# S2T Streaming API

A speech-to-text service built with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and FastAPI.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Vosk & Whisper (quick setup)

- Vosk (recommended for low-latency / realtime):

  ```bash
  pip install vosk
  # Download a Vosk model (see https://alphacephei.com/vosk/models)
  # e.g. unpack to ./models/vosk-model-small-en-in-0.4
  export VOSK_MODEL_PATH=./models/vosk-model-small-en-in-0.4
  ```

- Whisper / faster-whisper (file-based or higher-accuracy):

  ```bash
  pip install faster-whisper
  # Optionally choose model size when instantiating the service (main.py)
  # e.g. model_size="small" or "medium" for better accuracy.
  ```

- Note: Whisper cannot handle real time audio transcription like vosk so the output is very unstable

Place models where the server can access them and set `VOSK_MODEL_PATH` for Vosk before starting the app.

API is now live at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## Endpoints

### 1. `POST /transcribe` — Full Response

Uploads an audio file and **waits for the entire transcription to finish** before returning a single JSON response. Use this when you don't need results until processing is complete (e.g. server-side batch jobs, simple integrations).

**Request**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=en" \
  -F "word_timestamps=false"
```

**Response**
```json
{
  "language": "en",
  "language_probability": 0.9987,
  "duration": 12.34,
  "text": "Hello, this is the full transcription.",
  "segments": [
    { "start": 0.0, "end": 3.5, "text": "Hello, this is the full transcription." }
  ]
}
```

---

### 2. `POST /transcribe/stream` — SSE Streaming

Uploads the **same complete audio file** as endpoint 1, but instead of waiting for everything to finish, the server **streams each segment back to the client as soon as it is transcribed** using [Server-Sent Events (SSE)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events).

> **Key difference from `/transcribe`:** The audio file is still uploaded all at once — the streaming refers to how the *results* are delivered, not the audio. Segments appear in the frontend progressively as Whisper processes them, instead of all at once at the end.

**Request**
```bash
# -N disables curl's output buffering so you see events as they arrive
curl -N -X POST http://localhost:8000/transcribe/stream \
    -F "file=@audio.wav"
```

**Stream output** (one JSON event per line)
```
data: {"type": "info", "language": "en", "language_probability": 0.9987, "duration": 12.34}

data: {"type": "segment", "start": 0.0, "end": 3.5, "text": "Hello, this is the first segment."}

data: {"type": "segment", "start": 3.5, "end": 7.1, "text": "And this is the second one."}

data: {"type": "done"}
```

**Frontend (Flutter)**

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

final uri = Uri.parse('http://localhost:8000/transcribe/stream');
final request = http.MultipartRequest('POST', uri);

// Add audio file
request.files.add(await http.MultipartFile.fromPath('file', 'path/to/audio.wav'));

final response = await request.send();

response.stream.transform(utf8.decoder).split('\n').listen((line) {
    if (!line.startsWith('data: ')) return;
    
    final event = jsonDecode(line.substring(6));

    if (event['type'] == 'segment') print(event['text']);
    if (event['type'] == 'done') print('Transcription complete');
});
```

---

### 3. `WS /ws/transcribe` — WebSocket (Live / Chunked Audio)

A persistent WebSocket connection where the **client streams raw audio bytes in real time** (e.g. from a microphone), and the server transcribes whenever the client signals it is done speaking.

> **Key difference:** Unlike the HTTP endpoints where the whole file is sent upfront, here audio is sent as a continuous stream of binary chunks. This is ideal for live microphone recording in the browser — you send chunks as they are captured and trigger transcription when the user pauses or stops.

#### Protocol

| Direction | Message | Meaning |
|---|---|---|
| Client → Server | Binary frame | A chunk of raw audio bytes (any format: wav, webm, ogg…) |
| Client → Server | Text `"DONE"` | Stop buffering; transcribe everything received so far |
| Client → Server | Text `"RESET"` | Discard the buffer without transcribing |
| Server → Client | `{"type": "info", ...}` | Language detection result |
| Server → Client | `{"type": "segment", ...}` | One transcription segment |
| Server → Client | `{"type": "done"}` | Transcription finished for this utterance |
| Server → Client | `{"type": "reset"}` | Buffer cleared confirmation |
| Server → Client | `{"type": "error", "message": "..."}` | Something went wrong |

The connection **stays open** after a `done` event. The cycle repeats — the client can immediately start sending the next utterance's audio chunks and send `"DONE"` again.

#### Partial (incremental) results

The server **does not wait for `"DONE"` to start transcribing**. Every time a binary chunk arrives, a transcription pass is immediately triggered on the entire accumulated audio so far. Segments are streamed back to the client as soon as faster-whisper produces them.

To avoid duplicates across passes, the server tracks a `last_emitted_end` cursor and only forwards segments whose start time is beyond it.

An `asyncio.Lock` ensures passes never overlap — if a transcription is already running when a new chunk arrives, that chunk is silently accumulated and picked up by the next available pass.

#### When does it stop reading audio?

The server keeps reading chunks indefinitely. `"DONE"` finalises the utterance: the server waits for any in-flight pass, emits any remaining segments, sends `{"type": "done"}`, then resets the buffer and cursor for the next utterance. It does not auto-detect silence — your client decides when to send `"DONE"` (e.g. stop button, VAD on the frontend, fixed timer).


```dart
import 'package:web_socket_channel/web_socket_channel.dart';
import 'dart:convert';

final ws = WebSocketChannel.connect(
    Uri.parse('ws://localhost:8000/ws/transcribe'),
);

// --- Start recording and send chunks ---
final audioRecorder = AudioRecorder();
await audioRecorder.start(
    const RecordConfig(encoder: AudioEncoder.pcm16),
);

audioRecorder.onFrameReceived = (frame) {
    if (ws.sink != null) {
        ws.sink.add(frame.bytes);  // stream binary audio chunks
    }
};

// --- When user stops speaking ---
stopButton.onPressed = () async {
    await audioRecorder.stop();
    ws.sink.add('DONE');  // trigger transcription
};

// --- Receive results ---
ws.stream.listen((message) {
    final msg = jsonDecode(message);

    if (msg['type'] == 'info') {
        print('Detected language: ${msg['language']}');
    }
    if (msg['type'] == 'segment') {
        print('${msg['start']} → ${msg['end']} ${msg['text']}');
    }
    if (msg['type'] == 'done') {
        print('--- end of utterance ---');
    }
});
```

---

## Query Parameters (all endpoints)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `language` | string | `null` | BCP-47 code (`en`, `fr`, `de`…). Auto-detects if omitted. |
| `word_timestamps` | bool | `false` | Include per-word start/end times and confidence in the response. |

**Example with parameters**
```bash
curl -N -X POST "http://localhost:8000/transcribe/stream?language=en&word_timestamps=true" \
  -F "file=@audio.wav"
```

When `word_timestamps=true`, each segment includes a `words` array:
```json
{
  "type": "segment",
  "start": 0.0,
  "end": 2.4,
  "text": "Hello world",
  "words": [
    { "word": "Hello", "start": 0.0, "end": 0.6, "probability": 0.998 },
    { "word": "world", "start": 0.7, "end": 1.1, "probability": 0.995 }
  ]
}
```

---

## Choosing the Right Endpoint

| Scenario | Endpoint |
|---|---|
| Server-side batch processing | `POST /transcribe` |
| Upload a file, show results progressively in the UI | `POST /transcribe/stream` |
| Live microphone transcription | `WS /ws/transcribe` |

---
