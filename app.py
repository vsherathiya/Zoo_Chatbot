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
#    try:
#         app.run()
#    except Exception as e:
#         print("Failed in script at: " + str(e))

# from flask import jsonify, Flask, request
# from flask_cors import CORS, cross_origin
# import concurrent.futures  # Add this import
# from run_localGPT import main

# app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

# executor = concurrent.futures.ThreadPoolExecutor() 
# # Create a ThreadPoolExecutor for parallel processing


# @app.route("/ChatBotAPI/get-response-from-chatbot", methods=['POST'])
# @cross_origin()
# def get_response_from_chatbot():
#     data = request.get_json()
#     user_input = data["message"]
#     print(user_input)

#     # Use the ThreadPoolExecutor to process requests in parallel
#     future = executor.submit(main, user_input)
#     result = future.result()

#     return jsonify(result)


# if __name__ == '__main__':
#     try:
#         app.run(host='0.0.0.0', port=8000, threaded=True)  # Enable threaded mode for handling multiple requests simultaneously
#     except Exception as e:
#         print("Failed in script at: " + str(e))



# from flask import Flask, request, jsonify
# from flask_cors import CORS, cross_origin
# from run_localGPT import main
# import asyncio

# app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

# async def process_request(user_input):
#     # Simulate processing with asyncio.sleep, replace with your actual processing logic
#     await asyncio.sleep(5)
#     return user_input  # Return the result

# @app.route("/ChatBotAPI/get-response-from-chatbot", methods=['POST'])
# @cross_origin()
# async def get_response_from_chatbot():
#     data = request.get_json()
#     user_input = data["message"]
#     print(user_input)

#     # Use asyncio to process requests in parallel
#     result = await process_request(user_input)

#     return jsonify({"result": result})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)


# from flask import Flask, request, jsonify
# from flask_cors import CORS, cross_origin
# from concurrent.futures import ThreadPoolExecutor
# import time 

# app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'


# def process_request(user_input):
#     # Simulate processing, replace with your actual processing logic
#     time.sleep(5)
#     return user_input

# @app.route("/ChatBotAPI/get-response-from-chatbot", methods=['POST'])
# @cross_origin()
# def get_response_from_chatbot():
#     data = request.get_json()
#     user_input = data["message"]
#     print(user_input)

#     # Create a ThreadPoolExecutor to process requests in parallel
#     with ThreadPoolExecutor() as executor:
#         result = executor.submit(process_request, user_input)
#         result = result.result()  # Wait for the result

#     return jsonify({"result": result})


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)
# from flask import jsonify, Flask, request
# from concurrent.futures import ThreadPoolExecutor
# from run_localGPT import main

# app = Flask(__name__)

# # Create a ThreadPoolExecutor with a specified number of worker threads.
# executor = ThreadPoolExecutor(max_workers=4)  # Adjust the number of workers as needed.

# @app.route("/ChatBotAPI/get-response-from-chatbot", methods=['POST'])
# def get_response_from_chatbot():
#     data = request.get_json()
#     user_input = data["message"]
#     print(user_input)
#     # Use the ThreadPoolExecutor to execute the main function asynchronously.
#     future = executor.submit(main, user_input)
#     # You can also provide a timeout value for the response if needed.
#     # result = future.result(timeout=10)
    
#     return jsonify(future.result())

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)

# from flask import jsonify, Flask, request
# from flask_cors import CORS, cross_origin
# from concurrent.futures import ThreadPoolExecutor
# from run_localGPT import main

# app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

# # Create a ThreadPoolExecutor with a fixed number of threads
# executor = ThreadPoolExecutor(max_workers=5)

# def chatbot_response(user_input):
#     return main(user_input)

# @app.route("/ChatBotAPI/get-response-from-chatbot", methods=['POST'])
# @cross_origin()
# def get_response_from_chatbot():
#     data = request.get_json()
#     user_input = data["message"]
#     print(user_input)

#     # Use the ThreadPoolExecutor to submit the chatbot_response function
#     future = executor.submit(chatbot_response, user_input)
    
#     # Wait for the response and retrieve it when ready
#     chatbot_response = future.result()
    
#     return jsonify(chatbot_response})

# if __name__ == '__main__':
#     try:
#         app.run(host='0.0.0.0', port=8000)
#     except Exception as e:
#         print("Failed in script at: " + str(e))





from flask import jsonify, Flask, request
from flask_cors import CORS, cross_origin
from run_localGPT import main
import threading

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Create a lock to ensure thread safety
thread_lock = threading.Lock()

# Create an event to signal the response is ready
response_ready = threading.Event()

# Variable to store the chatbot response
chatbot_response = None

def chatbot_response(user_input):
    global chatbot_response
    with thread_lock:
        chatbot_response = main(user_input)
        response_ready.set()

@app.route("/ChatBotAPI/get-response-from-chatbot", methods=['POST'])
@cross_origin()
def get_response_from_chatbot():
    data = request.get_json()
    user_input = data["message"]
    print(user_input)
    
    # Create a new thread to handle the chatbot response
    response_thread = threading.Thread(target=chatbot_response, args=(user_input,))
    response_thread.start()
    
    # Wait for the response to be ready
    response_ready.wait()
    
    # Reset the event for the next request
    response_ready.clear()
    
    return jsonify({"message": chatbot_response})

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8000)
    except Exception as e:
        print("Failed in script at: " + str(e))
