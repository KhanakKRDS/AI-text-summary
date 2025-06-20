from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    print(">>> Hello route accessed")
    return "Flask is working!"

if __name__ == "__main__":
    app.run(port=5000, debug=True)
