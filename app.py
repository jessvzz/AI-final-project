from flask import Flask, render_template, request, redirect , url_for
from glob import glob
import csp_predictions
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/handle_data', methods=['POST'])
def handle_data():

    min_inv = float(request.form['min_inv'])
    max_inv = float(request.form['max_inv'])
    max_vol = float(request.form['max_vol'])

    bestreturnstocksset=csp_predictions.main(float(min_inv), float(max_inv), float(max_vol))
    print(bestreturnstocksset)
    return redirect(url_for('final_output', bst=bestreturnstocksset))


@app.route('/output')
def final_output():
    bst = request.args['bst']
    formatted_output = bst[1:-1].replace(", ", "\n")
    formatted_output = formatted_output.replace(".0", "$")
    formatted_output = formatted_output.replace(".csv", "")
    return render_template('output.html', formatted_output=formatted_output)
    #return f"Best stocks to invest in are <br>{formatted_output}"


if __name__ == '__main__':
    app.run(debug=True)