from flask import Flask, request, jsonify
from your_script import handle_form_submission

app = Flask(__name__)

@app.route('/submit', methods=['POST'])
def submit_form():
    form_data = request.form.to_dict()
    result = handle_form_submission(form_data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)