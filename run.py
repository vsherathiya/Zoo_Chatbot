from run_localGPT import main
while True:
    query = input("\nEnter a query: ")
    result = main(query)
    result = result['message']
    print(result)