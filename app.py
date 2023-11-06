# from flask import jsonify, Flask, render_template, request,send_from_directory
# from flask_cors import CORS, cross_origin
# from run_localGPT import main

# app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

# @app.route("/ChatBotAPI/get-response-from-chatbot", methods=['POST'])
# @cross_origin()
# def get_response_from_chatbot(): 
#     data = request.get_json()
#     user_input = data["message"]
#     print(user_input)
#     return  jsonify(main(user_input))

# if __name__ == '__main__':
#     try:
#         app.run(host='0.0.0.0', port=8000)
#     except Exception as e:
#         print("Failed in script at: " + str(e))






# from flask import jsonify, Flask, request
# from flask_cors import CORS, cross_origin
# from run_localGPT import main
# import threading

# app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

# # Create a lock to ensure thread safety
# thread_lock = threading.Lock()

# # Create an event to signal the response is ready
# response_ready = threading.Event()

# # Variable to store the chatbot response
# chatbot_response = None

# def chatbot_response(user_input):
#     global chatbot_response
#     with thread_lock:
#         chatbot_response = main(user_input)
#         response_ready.set()

# @app.route("/ChatBotAPI/get-response-from-chatbot", methods=['POST'])
# @cross_origin()
# def get_response_from_chatbot():
#     data = request.get_json()
#     user_input = data["message"]
#     print(user_input)
    
#     # Create a new thread to handle the chatbot response
#     response_thread = threading.Thread(target=chatbot_response, args=(user_input,))
#     response_thread.start()
    
#     # Wait for the response to be ready
#     response_ready.wait()
    
#     # Reset the event for the next request
#     response_ready.clear()
    
#     return jsonify({"message": chatbot_response})

# if __name__ == '__main__':
#     try:
#         app.run(host='0.0.0.0', port=8000)
#     except Exception as e:
#         print("Failed in script at: " + str(e))




# from flask import jsonify, Flask, request
# from flask_cors import CORS, cross_origin
# from run_localGPT import main
# import threading

# app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

# # Create a lock to ensure thread safety
# thread_lock = threading.Lock()

# # Create an event to signal the response is ready
# response_ready = threading.Event()

# # Variable to store the chatbot response
# chatbot_response = None


# def chatbot_response(user_input):
#     global chatbot_response
#     with thread_lock:
#         chatbot_response = main(user_input)
#         response_ready.set()

# @app.route("/ChatBotAPI/get-response-from-chatbot", methods=['POST'])
# @cross_origin()
# def get_response_from_chatbot():
#     data = request.get_json()
#     user_input = data["message"]
#     print(user_input)
    
#     # Create a new thread to handle the chatbot response
#     response_thread = threading.Thread(target=chatbot_response, args=(user_input,))
#     response_thread.start()
    
#     # Wait for the response to be ready
#     response_ready.wait()
    
#     # Reset the event for the next request
#     response_ready.clear()
    
#     return jsonify(chatbot_response)

# if __name__ == '__main__':
#     try:
#         app.run(host='0.0.0.0', port=8001)
#     except Exception as e:
#         print("Failed in script at: " + str(e))


from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from run_localGPT import main
import threading

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Create a lock to ensure thread safety
thread_lock = threading.Lock()

@app.route("/ChatBotAPI/get-response-from-chatbot", methods=['POST'])
@cross_origin()
def get_response_from_chatbot():
    data = request.get_json()
    user_input = data["message"]
    print(user_input)
    
    # Create a new thread to handle the chatbot response
    response_thread = threading.Thread(target=process_chatbot_response, args=(user_input,))
    response_thread.start()
    
    return "Request in progress..."

def process_chatbot_response(user_input):
    with thread_lock:
        chatbot_response = main(user_input)
    return jsonify({"message": chatbot_response})

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8001)
    except Exception as e:
        print("Failed in script at: " + str(e))
