from flask import jsonify, Flask, render_template, request,send_from_directory
from flask_cors import CORS, cross_origin
from run_localGPT import main
import time
from datetime import datetime
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/ChatBotAPI/get-response-from-chatbot", methods=['POST'])
@cross_origin()
def get_response_from_chatbot(): 
    data = request.get_json()
    user_input = data["message"]
    user_name = data["user_name"]
    print("--------------->User Intput >>>>>\n\n\n",user_input,"<<<<<<<<\n\n\n")
    start_time = time.time()
    response = main(user_input)
    end_time = time.time()
    start_datetime = datetime.fromtimestamp(start_time)
    end_datetime = datetime.fromtimestamp(end_time)
    response.update({'user_name':user_name})
    response.update({'strat_time': start_datetime.strftime('%Y-%m-%d %H:%M:%S')})
    response.update({"end_time":end_datetime.strftime('%Y-%m-%d %H:%M:%S')})
    response.update({"time_diff":str(start_time-end_time)})
    print("--------------->User Intput >>>>>\n\n\n",response,"<<<<<<<<\n\n\n")
    print(user_input)
    
    
    

    # Print the formatted datetime

    return  jsonify(response)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8000)
    except Exception as e:
        print("Failed in script at: " + str(e))
