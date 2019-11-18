
# IN CMD FIRST ENTER
# set FLASK_APP=WebApp.py
# NEXT ENTER 
# flask run
from flask import Flask, escape, request

app = Flask(__name__)

@app.route('/')
def hello():
    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}!'