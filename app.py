from flask import jsonify, Flask, render_template, request,send_from_directory
from flask_cors import CORS, cross_origin
from run_localGPT import main

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/ChatBotAPI/get-response-from-chatbot", methods=['POST'])
@cross_origin()
def get_response_from_chatbot(): 
    data = request.get_json()
    user_input = data["message"]
    print(user_input)
    return  jsonify(main(user_input))

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8000)
    except Exception as e:
        print("Failed in script at: " + str(e))