# Chatterbox TTS API

A quick API experiment to test returning audio data over HTTP.

Purpose: API backend for the Chatterbox TTS model.

## API Documentation

### Generate audio

Generates an MP3 file based on input.

__Requirement__: Currently requires an audio file called preview.mp3 to be in the root of the application.

```bash
# Generate MP3 Audio
POST /generate (or your specific endpoint)
```

### Health Check

Returns the current status of the API to ensure the service is reachable.

```bash
# Check Health
GET /health (or your specific endpoint)
```

### API Docs

Read the docs

```bash
GET /docs
```

