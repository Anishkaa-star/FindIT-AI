
import torch
import threading
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

from flask import Flask, render_template, request, jsonify
import sqlite3
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CLIP model settings
MODEL_NAME = "openai/clip-vit-base-patch32"

model = None
processor = None

device = "mps" if torch.backends.mps.is_available() else "cpu"


clip_load_lock = threading.Lock()
def load_clip():
    global model, processor

    if model is not None and processor is not None:
        return

    with clip_load_lock:
        if model is None or processor is None:
            print("Loading CLIP AI...")

            loaded_model = CLIPModel.from_pretrained(MODEL_NAME)
            loaded_processor = CLIPProcessor.from_pretrained(MODEL_NAME)

            loaded_model.to(device)
            loaded_model.eval()

            model = loaded_model
            processor = loaded_processor

            print("CLIP AI is ready!")


def get_image_embedding(image_path):
    load_clip()

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model.vision_model(
            pixel_values=pixel_values
        )

        features = model.visual_projection(
            outputs.pooler_output
        )

        features = torch.nn.functional.normalize(
            features, p=2, dim=-1
        )

    return features[0]


def find_matches(lost_image_path):
    lost_features = get_image_embedding(lost_image_path)

    conn = get_db()

    found_items = conn.execute("""
        SELECT * FROM items
        WHERE item_type = 'found'
        AND image IS NOT NULL
    """).fetchall()

    conn.close()

    matches = []

    for item in found_items:

        found_image_path = os.path.join(
            UPLOAD_FOLDER,
            item["image"]
        )

        if not os.path.exists(found_image_path):
            continue

        try:
            found_features = get_image_embedding(
                found_image_path
            )

            similarity = torch.dot(
                lost_features, found_features
            ).item()

            matches.append({
                "id": item["id"],
                "name": item["name"],
                "location": item["location"],
                "description": item["description"],
                "image": item["image"],
                "similarity": round(similarity * 100, 2)
            })

        except Exception as error:
            print("Could not compare image:", error)

    matches.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return matches[:5]


def get_db():
    conn = sqlite3.connect("findit.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            location TEXT NOT NULL,
            description TEXT,
            item_type TEXT NOT NULL DEFAULT 'found',
            image TEXT
        )
    """)

    columns = [
        row["name"].lower()
        for row in conn.execute(
            "PRAGMA table_info(items)"
        ).fetchall()
    ]

    if "item_type" not in columns:
        conn.execute("""
            ALTER TABLE items
            ADD COLUMN item_type TEXT NOT NULL DEFAULT 'found'
        """)

    if "image" not in columns:
        conn.execute("""
            ALTER TABLE items
            ADD COLUMN image TEXT
        """)

    conn.commit()
    conn.close()


init_db()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/items", methods=["GET"])
def get_items():

    conn = get_db()

    items = conn.execute(
        "SELECT * FROM items"
    ).fetchall()

    result = [dict(item) for item in items]

    conn.close()

    return jsonify(result)


@app.route("/api/items", methods=["POST"])
def add_item():

    name = request.form.get("name", "").strip()
    location = request.form.get("location", "").strip()
    description = request.form.get("description", "").strip()
    item_type = request.form.get(
        "item_type", ""
    ).strip().lower()

    if not name or not location:
        return jsonify({
            "error": "Name and location are required"
        }), 400

    if item_type not in ["lost", "found"]:
        return jsonify({
            "error": "Item type must be lost or found"
        }), 400

    image = request.files.get("image")
    image_filename = None

    if image and image.filename:

        if not allowed_file(image.filename):
            return jsonify({
                "error": "Only JPG, JPEG and PNG files are allowed"
            }), 400

        image_filename = secure_filename(image.filename)

        if not image_filename:
            return jsonify({
                "error": "Invalid filename"
            }), 400

        base, extension = os.path.splitext(image_filename)
        counter = 1

        while os.path.exists(
            os.path.join(
                app.config["UPLOAD_FOLDER"],
                image_filename
            )
        ):
            image_filename = f"{base}_{counter}{extension}"
            counter += 1

        image.save(
            os.path.join(
                app.config["UPLOAD_FOLDER"],
                image_filename
            )
        )

    conn = get_db()

    conn.execute("""
        INSERT INTO items
        (name, location, description, item_type, image)
        VALUES (?, ?, ?, ?, ?)
    """, (
        name,
        location,
        description,
        item_type,
        image_filename
    ))

    conn.commit()
    conn.close()

    matches = []

    # Automatically compare newly reported lost items
    if item_type == "lost" and image_filename:

        lost_image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            image_filename
        )

        try:
            matches = find_matches(lost_image_path)

        except Exception as error:
            print("AI matching error:", error)

    return jsonify({
        "message": "Item reported successfully!",
        "image": image_filename,
        "matches": matches
    }), 201


if __name__ == "__main__":
    app.run(debug=True, port=5001)
