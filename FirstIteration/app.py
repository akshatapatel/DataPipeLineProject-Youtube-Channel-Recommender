from flask import Flask, render_template 
# from flask.ext.cors import CORS, cross_origin

app = Flask(__name__)

# cors = CORS(app, resources={r"/": {"origins": "*"}})
# app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def home():
    return  render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
