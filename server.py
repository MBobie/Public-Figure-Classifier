import os
import json
import base64
import anthropic
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder=".")

CELEBRITIES = [
    "Noni Madueke", "Cristiano Ronaldo", "Donald Trump",
    "Kennedy Agyapong", "Seth Rollins", "Lyrical Joe",
    "Dwayne Johnson", "Lionel Messi", "Sarkodie",
    "Elon Musk", "Serwaa Amihere",
]

SYSTEM_PROMPT = f"""You are a Public Figure Classifier. Identify which of these people appears in the photo:
{json.dumps(CELEBRITIES)}

If you recognise one, return ONLY this JSON (no markdown, no extra text):
{{
  "class": "<exact name from the list>",
  "class_probability": [{{"name":"<name>","probability":<0-100>}}, ... one entry per person, all 11],
  "class_dictionary": {{{", ".join(f'"{n}": {i}' for i, n in enumerate(CELEBRITIES))}}}
}}

If no listed person is clearly visible, return:
{{"error": "Could not detect a recognised public figure."}}"""


def classify_with_claude(image_base64: str) -> list:
    """Call Claude Vision and return result list matching original API shape."""
    # strip data-uri prefix if present
    if "," in image_base64:
        header, data = image_base64.split(",", 1)
        media_type = header.split(":")[1].split(";")[0]
    else:
        data = image_base64
        media_type = "image/jpeg"

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": "Classify this image. Return JSON only."},
            ],
        }],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        raw = raw.replace("```", "").strip()

    parsed = json.loads(raw)

    if "error" in parsed:
        return [{"error": parsed["error"]}]

    # build class_probability as a flat list of probabilities in class_dictionary order
    prob_map = {item["name"]: item["probability"] for item in parsed.get("class_probability", [])}
    ordered_probs = [prob_map.get(name, 0.0) for name in CELEBRITIES]

    return [{
        "class": parsed["class"],
        "class_probability": ordered_probs,
        "class_dictionary": parsed["class_dictionary"],
    }]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "app.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)

@app.route("/classify_image", methods=["POST"])
def classify_image():
    image_data = request.form.get("image_data", "")
    if not image_data:
        return jsonify({"error": "No image data received."}), 400

    try:
        result = classify_with_claude(image_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    response = jsonify(result)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == "__main__":
    print("Starting Public Figure Classifier (Claude Vision backend)")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
