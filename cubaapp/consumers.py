# consumers.py
import json
import unittest
from channels.generic.websocket import WebsocketConsumer
from channels.generic.websocket import AsyncWebsocketConsumer
import asyncio
import json
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from django.shortcuts import render,redirect
from django.contrib import messages  
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout,authenticate
from django.contrib.auth.models import User
from .forms import *
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks.manager import AsyncCallbackManager
from django.core.serializers.json import DjangoJSONEncoder
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
import asyncio
import chardet
import json
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
import requests
import sys
from django.views.decorators.csrf import csrf_exempt
from asgiref.sync import sync_to_async, async_to_sync
from django.views.decorators.http import require_GET
from langchain.vectorstores.faiss import FAISS
from django.http import JsonResponse
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from pandasai import SmartDataframe
from langchain.agents import AgentExecutor
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import pandas as pd
from pandasai.llm import OpenAI
from django.conf import settings
from io import BytesIO
from .models import Chat, Imagen
import os 
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd 
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from langchain.chains.question_answering import load_qa_chain
from google.cloud import storage
from pandasai import SmartDataframe
from datetime import datetime
import numpy as np 
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI as OpenAI2
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from django.http import StreamingHttpResponse
import time
import csv
from io import StringIO
from django.views.decorators.http import condition
from typing import AsyncIterable
import traceback
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from django.views.decorators import gzip
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model="gpt-4-1106-preview", streaming=True, verbose=True, callbacks=[StreamingStdOutCallbackHandler()])
llm_dataframe = OpenAI(model="gpt-4-1106-preview")

chat_history = []
@tool
def obtener_respuesta(texto, archivo_url):
    """Retorna la respuesta a una pregunta sobre el dataset"""
    response_response = requests.get(archivo_url)
    result = chardet.detect(response_response.content)
    encoding = result['encoding']
    confidence = result['confidence']
    csv_data = StringIO(response_response.content.decode(encoding))
    dialect = csv.Sniffer().sniff(csv_data.read(1024))
    print(f"Detected Encoding: {encoding} with confidence: {confidence}, with delimiter: {dialect.delimiter}")
    df = pd.read_csv(archivo_url, encoding=encoding, delimiter=dialect.delimiter)
    df = SmartDataframe(df, config={"llm": llm_dataframe})
    return df.chat(texto)
tools = [obtener_respuesta]


MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un chatbot el cual te ayuda a responder preguntas o graficar en base a el dataset dado en español si se te pide una respuesta que contenga un dataframe o una tabla en general debes devolverlo como html y con las siguientes clases table table-striped table-bordered table-hover table-sm overflow-auto custom-table-class además de una breve explicación del dataframe, si es un gráfico devuelve el png.",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}, {archivo_url}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
def crear_agente():
    agent = (
                {
                    "input": lambda x: x["input"],
                    "agent_scratchpad": lambda x: format_to_openai_function_messages(
                        x["intermediate_steps"]
                    ),
                    "chat_history": lambda x: x["chat_history"],
                    "archivo_url": lambda x: x["archivo_url"],
                }
                | prompt
                | llm_with_tools
                | OpenAIFunctionsAgentOutputParser()
            )
    return agent
   

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

class DataStreamingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

        agent = AgentExecutor(agent=crear_agente(), tools=tools, verbose=True)
        message = 'Hola, cómo estás?'
        archivo_url = 'https://storage.googleapis.com/nuevodata/titanic.csv'

        async for chunk in agent.astream({"input": message, "archivo_url": archivo_url, "chat_history": chat_history}):
            await self.send(text_data=json.dumps({'chunk': chunk}))

class TestLangchainAsync(unittest.IsolatedAsyncioTestCase):
    async def test_aiter(self):
        handler = AsyncIteratorCallbackHandler()
        llm = OpenAI(
            temperature=0,
            streaming=True,
            callbacks=[handler],
            openai_api_key="",
        )
        print("Generating")
        prompt = PromptTemplate(
            input_variables=["product"],
            template="What is a good name for a company that makes {product}?",
        )
        prompt = prompt.format(product="colorful socks")
        asyncio.create_task(llm.agenerate([prompt]))
        print("Waiting for response")
        async for i in handler.aiter():
            print(i)