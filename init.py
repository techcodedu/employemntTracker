from flask import Flask

app = Flask(__name__)
app.secret_key = 'GliezelChester'
app.config['UPLOAD_FOLDER'] = 'uploads'

from routes import *  # Import routes
