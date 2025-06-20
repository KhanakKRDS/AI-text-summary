from flask import Flask, request, render_template
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "templates")

app = Flask(__name__, template_folder=template_dir)

@app.route("/", methods=["GET"])
def index():
    print(">>> Home page accessed")
    return render_template("AI-text-summary.html", og="", summary="")

@app.route("/process", methods=["POST"])
def process():
    print(">>> Process route hit")
    try:
        original_text = request.form.get("og", "").strip()
        print(f"Original text length: {len(original_text)}")
        summary = f"Summary would appear here for: {original_text[:30]}..."
        print("Summary generated")
    except Exception as e:
        summary = f"Error: {str(e)}"
        print(f"Exception occurred: {e}")
    return render_template("AI-text-summary.html", og=original_text, summary=summary)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=3000, debug=True)
