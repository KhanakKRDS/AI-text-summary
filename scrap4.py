from flask import Flask

# Create the Flask app
app = Flask(__name__)

# Define a basic route
@app.route("/")
def index():
    return "<h1>Flask is working!</h1>"

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
