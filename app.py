from flask import Flask, render_template, request, redirect , url_for
from glob import glob
import csp_predictions
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/handle_data', methods=['POST'])
def handle_data():

    # Get the data from the form
    min_inv = request.form['min_inv']
    max_inv = request.form['max_inv']
    max_vol = request.form['max_vol']
    bestreturnstocksset,returnexpected=csp_predictions.main(float(min_inv), float(max_inv), float(max_vol))
    print(bestreturnstocksset)
    print(returnexpected)
    return redirect (url_for('final_output', bst=bestreturnstocksset, ret=returnexpected))

@app.route('/output')
def final_output():
    bst=request.args['bst']
    ret=request.args['ret']
    return f"Best stocks to invest in are {bst} with an expected return of {ret}"

if __name__ == '__main__':
    app.run(debug=True)