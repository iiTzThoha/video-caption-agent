# Video Caption Agent

An AI agent that watches a video clip and generates captions in four styles: **formal**, **sarcastic**, **humorous_tech**, and **humorous_non_tech**.

Built for the AMD Developer Hackathon: Act II Track 2 (Video Captioning Agent).

## How it works

The pipeline uses a two-stage approach via the [Fireworks AI](https://fireworks.ai) API:

1. **Video understanding**: minimax-m3 watches the video directly (native video input) and produces a detailed factual description of the setting, subjects, actions, and mood.
2. **Style generation**: The description is rewritten into each requested caption style in a single API call, with intelligent key normalization to handle variations in model output.

## Optimizations

- **Single-pass generation**: Each video requires only 2 API calls (1 for description, 1 for styled captions)
- **Key normalization**: Automatically handles common misspellings (e.g., "sarcasm" → "sarcastic") without retries
- **Graceful failure**: If a task fails, empty captions are written and processing continues; the container still exits 0 as long as `results.json` was written successfully

## Docker image

Pulled from Docker Hub:
docker pull iitzthoha/video-caption-agent:latest

## Input / Output format

**Input** (`/input/tasks.json`):

```json
[
  {
    "task_id": "v1",
    "video_url": "https://example.com/clip1.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

**Output** (`/output/results.json`):

```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

## Running locally

```bash
docker build -t video-caption-agent .

docker run --rm \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  video-caption-agent
```

No API key needs to be passed at runtime, it's baked into the image at build time (see Requirements below). If you want to test with a different key without rebuilding, you can still override it:

```bash
docker run --rm \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  -e FIREWORKS_API_KEY=your_key_here \
  video-caption-agent
```

## Requirements

- `FIREWORKS_API_KEY` is baked into the image at build time via the Dockerfile (`ENV FIREWORKS_API_KEY=...`). Track 2 does not inject any credentials at evaluation time, so the container must carry its own key.
- No local GPU or model weights required. All inference happens via the Fireworks AI API.
- Video clips should be publicly accessible URLs (direct MP4 links).
- If running `pipeline.py` directly (outside Docker) for local debugging, install dependencies first: `pip install requests`.

## Error handling

If a task fails (e.g., unreachable video URL, API error), the pipeline logs the error, writes empty captions for that task, and continues processing the remaining tasks rather than crashing the whole run. The process exits with code 0 as long as `results.json` is written, so a single failed clip doesn't affect the rest of the submission.

## Example clips (for development)

- Urban: https://storage.googleapis.com/amd-hackathon-clips/1868079-uhd_2560_1440_25fps.mp4
- Kitten: https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4
- Office: https://storage.googleapis.com/amd-hackathon-clips/3044693-uhd_3840_2160_24fps.mp4
