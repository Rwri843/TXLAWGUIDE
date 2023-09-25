from flask import Flask, request, jsonify, render_template
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
from whitenoise import WhiteNoise
load_dotenv()

app = Flask(__name__)

app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')

port = int(os.environ.get('PORT', 5000))
app.run(host="0.0.0.0", port=port)

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
print("API Key from environment:", OPENAI_API_KEY)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.json.get('prompt', '')

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        title_template = PromptTemplate(
            input_variables=['topic'],
            template='Search the Texas Penal Code for the definition, elements of offense, and punishment{topic}'
        )

        script_template = PromptTemplate(
            input_variables=['title'],
            template='Texas Penal Code Title {title}'
        )

        llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.0)

        title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title')
        script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script')
        sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)

        response = sequential_chain({'topic': prompt})
        title = response.get('title', 'No title generated')
        script = response.get('script', 'No script generated')

        print("Backend response:", {'title': title, 'script': script})

        return jsonify({'title': title, 'script': script})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')
