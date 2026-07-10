# Video Caption Agent

An AI agent that watches a video clip and generates captions in four type of styles: **formal**, **sarcastic**, **humorous_tech**, and **humorous_non_tech**.

Built for the AMD Developer Hackathon: Act II Track 2 (Video Captioning Agent).

## How it works

The pipeline uses a two stage approach via the [Fireworks AI](https://fireworks.ai) API:

1. **Video understanding** : `minimax-m3` watches the video directly (native video input) and produces a detailed description of the setting, subjects, actions, and mood.
2. **Style generation** : the description is rewritten into each requested caption style in a single call, with a strict key validation and automatic retry if the model returns malformed output.

## Docker image

Pulled from Docker Hub:

​```
docker pull iitzthoha/video-caption-agent:latest
​```

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
  -e FIREWORKS_API_KEY=your_key_here \
  video-caption-agent
```

## Requirements

- `FIREWORKS_API_KEY` must be set as an environment variable at runtime (not baked into the image).
- No local GPU or model weights required. All inference happens via the Fireworks AI API.

## Error handling

If a task fails (e.g. an unreachable video URL), the pipeline logs the error, writes empty captions for that task, and continues processing the remaining tasks rather than crashing the whole run.
