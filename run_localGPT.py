import os
import logging
import click
import torch
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
import torch
torch.manual_seed(42)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)

from load_models import (
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from constants import *

# (
#     EMBEDDING_MODEL_NAME,
#     PERSIST_DIRECTORY,
#     MODEL_ID,
#     MODEL_BASENAME,
#     MAX_NEW_TOKENS,
#     MODELS_PATH,
#     CHROMA_SETTINGS,*
# )


def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.05,
        # top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm

if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"

logging.info(f"Running on: {DEVICE_TYPE}")
logging.info(f"Display Source Documents set to: {SHOW_SOURCE}")

EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})


DB = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=EMBEDDINGS,
    client_settings=CHROMA_SETTINGS,
)

RETRIEVER = DB.as_retriever()

LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
prompt, memory = get_prompt_template(promptTemplate_type="llama", history=False)

QA = RetrievalQA.from_chain_type(
    llm=LLM,
    chain_type="stuff",
    retriever=RETRIEVER,
    return_source_documents=SHOW_SOURCE,
    chain_type_kwargs={
        "prompt": prompt,
    },
)
    

def main(query,device_type="cuda" if torch.cuda.is_available() else "cpu",
         show_sources=SHOW_SOURCE, use_history=USE_HISTORY, model_type='mistral'):
    global DB
    global QA
    # print(QA)
    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)


    # retriever = db.as_retriever()
    res = QA(query)
    answer = res["result"]

    #docs_and_scores = db_for_similarity.similarity_search_with_relevance_scores(answer)
    docs_and_scores1 = DB.similarity_search_with_relevance_scores(query)
    #docs_and_scores2 = db_for_similarity.similarity_search_with_relevance_scores(docs_and_scores)
    #docs_and_scores1 = db_for_similarity.similarity_search_with_relevance_scores(docs_and_scores[0])
    if docs_and_scores1[0][1] < 0.67: # & docs_and_scores1[0][1] < 0.72:
        answer = "Ich weiß nicht"
        print("\n> similarity score of query:")
        print(docs_and_scores1[0][1])      
    elif 0.67 <= docs_and_scores1[0][1] <= 0.75:
        res = QA(query)
        Temp_answer, docs = res["result"], res["source_documents"]
        
        docs_and_scores2 = DB.similarity_search_with_relevance_scores(Temp_answer)
        if docs_and_scores2[0][1] < 0.75:# & docs_and_scores2[0][1] < 0.72:
            answer = "Ich weiß nicht"

            print("\n> similarity score of answer:")
            print(docs_and_scores2[0][1])
            print("\n> similarity score of query:")
            print(docs_and_scores1[0][1])               
        else:
            answer = Temp_answer
            print("\n> similarity score of answer:")
            print(docs_and_scores2[0][1])
            print("\n> similarity score of query:")
            print(docs_and_scores1[0][1]) 
    else:
        answer = answer
        
        print("\n> similarity score of query:")
        print(docs_and_scores1[0][1]) 
    print(docs_and_scores1[0][1])
    # Print the result
    print("\n\n> Question:")
    print(query)
    print("\n> Answer:")
    print(answer)
    if answer.lower() == "ich weiss nicht":
        answer = "Leider konnte unser System keine Antwort finden.\
            Wir entschuldigen uns dafür und unser Support wird sich mit Ihnen in \
                Verbindung setzen. Bitte teilen Sie Ihre E-Mail mit."
    return {"message": answer}
    # query = input("\nEnter a query: ")
    # Get the answer from the chain
    

