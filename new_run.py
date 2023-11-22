import os
import logging
import torch
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from prompt_template_utils import get_prompt_template
from langchain.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline
)
from load_models import (
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model
)
from constants import (SHOW_SOURCE,
                       MAX_NEW_TOKENS,
                       EMBEDDING_MODEL_NAME,
                       MODEL_BASENAME,
                       MODEL_ID,
                       CHROMA_SETTINGS,
                       PERSIST_DIRECTORY,
                       MODELS_PATH,
                       USE_HISTORY
                       )
import time

torch.manual_seed(42)
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(
                model_id,
                model_basename,
                device_type,
                LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(
                model_id,
                model_basename,
                device_type,
                LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(
                model_id,
                model_basename,
                device_type,
                LOGGING)
    else:
        model, tokenizer = load_full_model(
            model_id,
            model_basename,
            device_type,
            LOGGING)

    generation_config = GenerationConfig.from_pretrained(model_id)

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






def main(query, device_type="cuda" if torch.cuda.is_available() else "cpu",
         show_sources=SHOW_SOURCE, use_history=USE_HISTORY):
    
    EMBEDDINGS = HuggingFaceInstructEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": DEVICE_TYPE}
    )

    DB = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=EMBEDDINGS,
        client_settings=CHROMA_SETTINGS,
    )

    RETRIEVER = DB.as_retriever()
    prompt, memory = get_prompt_template(promptTemplate_type="mistral",
                                        history=False)
        
    LLM = load_model(device_type=DEVICE_TYPE,
                    model_id=MODEL_ID,
                    model_basename=MODEL_BASENAME
                    )

    QA = RetrievalQA.from_chain_type(
    llm=LLM,
    chain_type="stuff",
    retriever=RETRIEVER,
    return_source_documents=SHOW_SOURCE,
    chain_type_kwargs={
        "prompt": prompt,
    },
)
    # print(QA)
    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)
    start_time = time.time()
    res = QA(query)
    answer = res["result"]
    docs_and_scores1 = DB.similarity_search_with_relevance_scores(query)
    if docs_and_scores1[0][1] < 0.67:
        answer = "Ich weiß nicht"
        print("\n> similarity score of query:")
        print(docs_and_scores1[0][1])
    elif 0.67 <= docs_and_scores1[0][1] <= 0.75:
        res = QA(query)
        Temp_answer = res["result"]
        docs_and_scores2 = DB.similarity_search_with_relevance_scores(
            Temp_answer)
        if docs_and_scores2[0][1] < 0.75:
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
    time_diff = time.time() - start_time
    return {"message": answer + "\nTime :" + str(round(time_diff,2)) + " Seconds" }
