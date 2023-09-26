from flask import Flask
import pipelinetest

app = Flask(__name__)

@app.route('/')
def dynamic_page():
    print("runing dynamic_page()")
    return pipelinetest.get_result()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000', debug=True)