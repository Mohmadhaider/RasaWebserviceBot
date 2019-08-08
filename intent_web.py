from rasa_nlu.training_data  import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Metadata, Interpreter
from flask import Flask,flash, redirect, render_template, request, url_for

app = Flask(__name__)
app.config["DEBUG"] = True

train_data = load_data('rasa_dataset.json')
trainer = Trainer(config.load("config_spacy.yaml"))

trainer.train(train_data)
model_directory = trainer.persist('/projects')
interpreter = Interpreter.load(model_directory)

@app.route('/')
def home():
     if 'text' in request.args:
        txt = request.args['text']
        #id = int(request.args['text'])
        return interpreter.parse(txt)
     else:
        return "Please write any query."

app.run()
