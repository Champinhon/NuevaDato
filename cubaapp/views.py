import threading
from django.shortcuts import render,redirect
from django.contrib import messages  
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout,authenticate
from django.contrib.auth.models import User
from .forms import *
import asyncio
import chardet
import openai
from queue import Queue
from typing import Any
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
import requests
from asgiref.sync import sync_to_async, async_to_sync
from django.views.decorators.http import require_GET
from langchain.vectorstores.faiss import FAISS
from django.http import JsonResponse
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import tool
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
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.schema import LLMResult
import time
import csv
from io import StringIO
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model="gpt-4-1106-preview", streaming=True, verbose=True, callbacks=[StreamingStdOutCallbackHandler()])

llm_dataframe = OpenAI(model="gpt-4-1106-preview")

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

def get_response_pdf(request, prompt_question, pdf_url):
    if request.method == 'POST':
        response = requests.get(pdf_url)
        if response.status_code == 200:
            pdf_data = BytesIO(response.content)

            reader = PdfReader(pdf_data)
            raw_text = ''
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text
            textsplitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            texts = textsplitter.split_text(raw_text)
            embeddings = OpenAIEmbeddings()
            docsearch = FAISS.from_texts(texts, embeddings)
            chain = load_qa_chain(OpenAI2(), chain_type='stuff')
            docs = docsearch.similarity_search(prompt_question)
            response_text = chain.run(input_documents=docs, question=prompt_question)
            print(response_text, 'response_text')
            return response_text
        else:
            return JsonResponse({'error': f'Error al descargar el archivo: {response.status_code}'}, status=400)

    return JsonResponse({'error': 'Método HTTP no permitido'}, status=405)

def get_latest_chats(request):
    chats = Chat.objects.filter(user=request.user)
    latest_chats = [{'message': chat.message, 'response': chat.response} for chat in chats]
    return JsonResponse({'chats': latest_chats})

def obtener_hora_actual():
    ahora = datetime.now()
    hora_actual = ahora.strftime("%H:%M:%S")
    return hora_actual

# Ejemplo de uso
hora_actual = obtener_hora_actual()
print(f"Hora actual con segundos: {hora_actual}")
def download_csv_from_url(url, local_filename):
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(local_filename, 'wb') as csv_file:
            csv_file.write(response.content)

    return redirect('index')

chat_history = []

def upload_file(request):
    if request.method == 'POST':
        form = ArchivoForm(request.POST, request.FILES)
        if form.is_valid():
            archivo = form.save(commit=False)
            client = storage.Client(project=settings.GS_PROJECT_ID, credentials=settings.GS_CREDENTIALS)
            filename = archivo.archivo.name
            bucket = client.bucket(settings.GS_BUCKET_NAME)
            blob = bucket.blob(filename)
            blob.upload_from_file(archivo.archivo.file)
            archivo.archivo_url = f'https://storage.googleapis.com/{settings.GS_BUCKET_NAME}/{filename}'
            archivo.save()
            request.session['archivo_url'] = archivo.archivo_url
            print(archivo.archivo_url)
            return redirect('index')

    else:
        form = ArchivoForm()

    return render(request, 'applications/chat/chat/upload_file.html', {'form': form})

@login_required(login_url="/login_home")
def index(request):
    chats = Chat.objects.filter(user=request.user)
    archivo_url = request.session.get('archivo_url')
    uploaded_filename = None
    if archivo_url:
        uploaded_filename = os.path.basename(archivo_url)
    
    if request.method == 'POST':
        message = request.POST.get('message')
        archivo_url = request.session.get('archivo_url')
        if 'xls' in str(archivo_url):
            excel_path = archivo_url
            csv_path = f'temp_file_{timezone.now().strftime("%Y%m%d%H%M%S")}.csv'
            try:
                df = pd.read_excel(excel_path)

                df.to_csv(csv_path, index=False, encoding='utf-8')

                client = storage.Client(project=settings.GS_PROJECT_ID, credentials=settings.GS_CREDENTIALS)
                bucket = client.bucket(settings.GS_BUCKET_NAME)
                filename = os.path.basename(csv_path)
                blob = bucket.blob(filename)
                blob.upload_from_filename(csv_path)

                csv_url = f'https://storage.googleapis.com/{settings.GS_BUCKET_NAME}/{filename}'

                # Eliminar el archivo temporal CSV local
                os.remove(csv_path)

                # Resto del código ...
            except Exception as e:
                print(f"Error al procesar archivo Excel: {e}")
                return f"Error al procesar archivo Excel: {e}"
            agent_executor = AgentExecutor(agent=crear_agente(), tools=tools, verbose=True)
            result = agent_executor.invoke({"input": message, "archivo_url": csv_url, "chat_history": chat_history})
            chat_history.extend(
                [
                    HumanMessage(content=message),
                    AIMessage(content=result["output"]),
                ]
            )
            
            print(result, 'result')
            response_data = result["output"]
            if 'png' in str(response_data):
                local_image_path = 'exports/charts/temp_chart.png'
                print(local_image_path, 'local_image_path')
                if os.path.exists(local_image_path):
                    client = storage.Client(project=settings.GS_PROJECT_ID, credentials=settings.GS_CREDENTIALS)
                    bucket = client.bucket(settings.GS_BUCKET_NAME)
                    filename = f'grafico_{request.user.username}_{timezone.now().strftime("%Y%m%d%H%M%S")}.png'
                    blob = bucket.blob(filename)

                    with open(local_image_path, 'rb') as image_file:
                        blob.upload_from_file(image_file, content_type='image/png')

                    image_url = f'https://storage.googleapis.com/{settings.GS_BUCKET_NAME}/{filename}'
                    image_instance = Imagen.objects.create(usuario=request.user, url=image_url)
                    image_instance.save()
                    response = image_url
                    os.remove(local_image_path)
                    response = image_url
            else:
                response = response_data
            chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
            chat.save()
            return JsonResponse({'response': response})
        elif 'pdf' in str(archivo_url):
            response = get_response_pdf(request, message, archivo_url)
            print(response, 'response')
            chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
            chat.save()
            print(response, 'chat')
            return JsonResponse({'response': response})
        elif 'csv' in str(archivo_url):
            agent_executor = AgentExecutor(agent=crear_agente(), tools=tools, verbose=True)
            print(chat_history, 'chat_history')
            result = agent_executor.invoke({"input": message, "archivo_url": archivo_url, "chat_history": chat_history})  
            print(result, 'result')
            chat_history.extend(
                    [
                        HumanMessage(content=message),
                        AIMessage(content=result["output"]),
                    ]
                )
            response_data = result["output"]
            if 'png' in str(response_data):
                local_image_path = 'exports/charts/temp_chart.png'
                print(local_image_path, 'local_image_path')
                if os.path.exists(local_image_path):
                    client = storage.Client(project=settings.GS_PROJECT_ID, credentials=settings.GS_CREDENTIALS)
                    bucket = client.bucket(settings.GS_BUCKET_NAME)
                    filename = f'grafico_{request.user.username}_{timezone.now().strftime("%Y%m%d%H%M%S")}.png'
                    blob = bucket.blob(filename)
                    with open(local_image_path, 'rb') as image_file:
                        blob.upload_from_file(image_file, content_type='image/png')
                    image_url = f'https://storage.googleapis.com/{settings.GS_BUCKET_NAME}/{filename}'
                    image_instance = Imagen.objects.create(usuario=request.user, url=image_url)
                    image_instance.save()
                    response = image_url
                    os.remove(local_image_path)
                    response = image_url
            else:
                response = response_data
                
            chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
            chat.save()
            return JsonResponse({'response': response})
    return render(request, 'applications/chat/chat/chat.html', {'chats': chats, 'archivo_url': uploaded_filename})

def login_home(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
       
        
        if form.is_valid():
       
          username=form.cleaned_data.get('username')
          password=form.cleaned_data.get('password')
          user=authenticate(username=username,password=password)
          if user is not None:
            login(request,user)
            return redirect("index")
          else:
            messages.error(request,"Wrong credentials")
            return redirect("login_home")
        else:
            messages.error(request,"Wrong credentials")
            return redirect("login_home")

    else:
        form =AuthenticationForm()

       
    return render(request,'main-login.html',{"form":form,})


def logout_view(request):
    logout(request)
    return redirect('login_home')



def signup_home(request):
    if request.method=='GET':
        return render(request,'signup.html') 
    else:
        email=request.POST['email']
        username=request.POST['username']
        password=request.POST['password']
        user=User.objects.filter(email=email).exists()
        if user:  
            raise Exception('Something went wrong')
        new_user=User.objects.create_user(username=username,email=email,password=password)
        new_user.save()
        return redirect('index')

def plan_economico(request):
    return render(request,'applications/chat/chat/pagos_economico.html')
