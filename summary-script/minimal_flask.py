from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    print("Home page accessed")
    return "Hello Flask is running!"

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=3000, debug=True)
