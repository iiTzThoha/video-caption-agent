import json
import os
import sys
import requests

BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL = "accounts/fireworks/models/minimax-m3"

STYLE_DEFINITIONS = {
    "formal": "Professional, objective, factual tone",
    "sarcastic": "Dry, ironic, lightly mocking",
    "humorous_tech": "Funny, with technology or programming references",
    "humorous_non_tech": "Funny, everyday humour with no technical jargon",
}

INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"


def get_api_key():
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        print("ERROR: FIREWORKS_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    return key


def normalize_key(key):
    """Convert various spellings to our standard keys"""
    key_lower = key.lower().strip()

    key_mappings = {
        'sarcastic': 'sarcastic',
        'sarcasm': 'sarcastic',
        'sarcastik': 'sarcastic',
        'sarcastc': 'sarcastic',
        'humorous_tech': 'humorous_tech',
        'tech_humour': 'humorous_tech',
        'tech_humor': 'humorous_tech',
        'humorous_non_tech': 'humorous_non_tech',
        'nontech_humour': 'humorous_non_tech',
        'nontech_humor': 'humorous_non_tech',
        'formal': 'formal',
        'formel': 'formal',
    }

    if key_lower in key_mappings:
        return key_mappings[key_lower]

    if 'sarcas' in key_lower:
        return 'sarcastic'

    return key_lower


def get_video_description(video_url, api_key):
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this video in detail: the setting, subjects, "
                                "actions, and mood. Be factual and specific. 3-5 sentences."
                            ),
                        },
                        {"type": "video_url", "video_url": {"url": video_url}},
                    ],
                }
            ],
            "max_tokens": 5000,
        },
        timeout=25,
    )
    resp.raise_for_status()
    result = resp.json()
    finish_reason = result["choices"][0]["finish_reason"]
    if finish_reason == "length":
        print(f"WARNING: description cut off for {video_url}", file=sys.stderr)
    return result["choices"][0]["message"]["content"]


def _map_parsed_to_styles(parsed, styles):
    """Match parsed JSON keys to requested styles, using normalization as a fallback."""
    result = {}
    for style in styles:
        if style in parsed:
            result[style] = parsed[style]
            continue
        found = False
        for key, value in parsed.items():
            if normalize_key(key) == style:
                result[style] = value
                found = True
                break
        if not found:
            result[style] = ""
    return result


def get_styled_captions(description, styles, api_key, max_retries=1):
    """Generate styled captions. Tries up to (max_retries + 1) times total,
    only retrying if keys are missing after normalization. Bounded so a bad
    response can't spiral into a long stall."""
    style_list = ", ".join(styles)
    definitions_text = "\n".join(f"- {s}: {STYLE_DEFINITIONS[s]}" for s in styles)
    keys_example = ", ".join(f'"{s}": "..."' for s in styles)

    prompt = f"""Here is a factual description of a video:

"{description}"

Write a caption for this video in EACH of the following styles: {style_list}

Style definitions:
{definitions_text}

CRITICAL: Respond with ONLY a valid JSON object, no other text.
You MUST use these EXACT key names, spelled exactly as shown, nothing else:
{{{keys_example}}}

Do not misspell the keys. Copy them exactly as given above."""

    result = {s: "" for s in styles}

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 5000,
                },
                timeout=25,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            parsed = json.loads(content)
        except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"WARNING: attempt {attempt + 1} failed: {e}", file=sys.stderr)
            continue

        result = _map_parsed_to_styles(parsed, styles)
        missing = [s for s in styles if not result[s]]
        if not missing:
            return result  # all styles present, done

        print(f"WARNING: attempt {attempt + 1}: missing styles {missing}", file=sys.stderr)
        # loop again only if we have retries left; otherwise fall through and
        # return whatever we have (partial credit beats a crashed task)

    return result


def process_task(task, api_key):
    task_id = task["task_id"]
    video_url = task["video_url"]
    styles = task["styles"]

    print(f"Processing {task_id}: {video_url}", file=sys.stderr)

    description = get_video_description(video_url, api_key)
    captions = get_styled_captions(description, styles, api_key)

    return {"task_id": task_id, "captions": captions}


def main():
    api_key = get_api_key()

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    results = []
    had_failure = False

    for task in tasks:
        try:
            result = process_task(task, api_key)
            results.append(result)
        except Exception as e:
            print(f"ERROR processing task {task.get('task_id', '?')}: {e}", file=sys.stderr)
            had_failure = True
            results.append({
                "task_id": task.get("task_id", "unknown"),
                "captions": {s: "" for s in task.get("styles", [])},
            })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done. Wrote {len(results)} results to {OUTPUT_PATH}", file=sys.stderr)
    sys.exit(1 if had_failure else 0)


if __name__ == "__main__":
    main()
