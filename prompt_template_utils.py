"""
This file implements prompt template for llama based models. 
Modify the prompt template based on the model you select. 
This seems to have significant impact on the output of the LLM.
"""

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# this is specific to Llama-2.

# system_prompt = """You're a customer support bot that is talking to a user, you will use the provided context to answer user questions. Conversation between you and user should be like human and your reply should be in german language ,Only Language You Should Understand is German dont provide Other Language answer,read the given context before answering questions and think step by step. Your answer should be short, precise, and accurate and in german language. Furthermore, Your answer should be Straight foreword and in german. If you can not answer a user question based on the provided context, inform the user. Do not use any other information for answering user. Make sure that you don't provide any explanation, reason or any suggestion when you don't have user answer from provided context."""


system_prompt = """You're a customer support bot that is talking to a user, you will use the provided context to answer user questions. Conversation between you and user should be like human and your reply should be always in german language ,Only Language You Should Understand is German dont provide Other Language answer, read the given context before answering questions and think step by step. Your answer should be short, precise, and accurate and in german language. Furthermore, Your answer should be Straight foreword and in german. If you can not answer a user question based on the provided context, inform the user. Do not use any other information for answering user. Make sure that you don't provide any explanation, reason or any suggestion when you don't have user answer from provided context. If Question asked is in English language than your reply should be  "Bitte stellen Sie Ihre Frage in deutscher Sprache"."""


def get_prompt_template(system_prompt=system_prompt, promptTemplate_type=None, history=False):
    if promptTemplate_type == "llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}
            Answer in the following language: German
            """

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}
            
            Answer in the following language: German """

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    elif promptTemplate_type == "mistral":
        B_INST, E_INST = "<s>[INST] ", " [/INST]"
        if history:
            prompt_template = (
                B_INST
                + system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}
            Answer in the following language: German"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                B_INST
                + system_prompt
                + """
            
            Context: {context}
            User: {question}
            Answer in the following language: German"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    else:
        # change this based on the model you have selected.
        if history:
            prompt_template = (
                system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}
            Answer in the following language: German Deutsch
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                system_prompt
                + """
            
            Context: {context}
            User: {question}
            Answer in the following language: German Deutsch
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return (
        prompt,
        memory,
    )
