
from flask import Flask, render_template, request, redirect, url_for
import csp_predictions
from webbrowser import open
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/handle_data', methods=['POST'])
def handle_data():
    try:
        min_inv = float(request.form['min_inv'])
        max_inv = float(request.form['max_inv'])
        max_vol = float(request.form['max_vol'])


        bestreturnstocksset = csp_predictions.main(min_inv, max_inv, max_vol)

        if bestreturnstocksset is None:
            if max_inv <= 100:
                error_message = "We can't produce an optimized portfolio for these values, please try multiples of 10"
            elif max_inv > 100 and max_inv <= 400:
                error_message = "We can't produce an optimized portfolio for these values, please try multiples of 50"
            else:
                error_message = "We can't produce an optimized portfolio for these values, please try multiples of 100"

            return render_template('index.html', error_message=error_message)


        return redirect(url_for('final_output', bst=bestreturnstocksset))
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('index.html', error_message=error_message)





@app.route('/output')
def final_output():
    bst = request.args['bst']
    formatted_output = bst[1:-1].replace(", ", "\n")
    formatted_output = formatted_output.replace(".0", "$")
    formatted_output = formatted_output.replace(".csv", "")

    return render_template('output.html', formatted_output=formatted_output)


if __name__ == '__main__':
    open('http://localhost:5000/')
    app.run(debug=True)