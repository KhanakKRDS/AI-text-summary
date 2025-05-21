from flask import Flask, request, render_template

app = Flask(__name__, template_folder='C:\Khanak\AI-text-summary')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def submit():
    Number1 = int(request.form.get('num1'))
    Number2 = int(request.form.get('num2'))
    
    sum = Number1 + Number2
    
    return render_template('index.html', sum=sum)

if __name__ == '__main__':
    app.run(debug=True)
