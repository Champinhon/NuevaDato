#Django
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from .forms import ArchivoForm
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse

#Python
import os
import requests
from io import BytesIO, StringIO
import json

#Terceros
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import openai
import pandas as pd 
from langchain.chains.question_answering import load_qa_chain
from google.cloud import storage
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent

openai.api_key = "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"

os.environ["OPENAI_API_KEY"] = "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"
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
            chain = load_qa_chain(OpenAI(), chain_type='stuff')
            docs = docsearch.similarity_search(prompt_question)

            response_text = chain.run(input_documents=docs, question=prompt_question)

            return JsonResponse({'response': response_text})
        else:
            return JsonResponse({'error': f'Error al descargar el archivo: {response.status_code}'}, status=400)

    return JsonResponse({'error': 'Método HTTP no permitido'}, status=405)

def get_response_csv(request, prompt_question, csv_url):
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        csv_url,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    respuesta = agent.run(prompt_question) 
    if respuesta:
        return respuesta
    else:
        print("ERROR.")

def get_response_xlsx(request, prompt_question, xls_url):
    response = requests.get(xls_url)
    if response.status_code == 200:
        xls_data = BytesIO(response.content)
        
        df = pd.read_excel(xls_data)
        csv_path = 'temp_file.csv'
        df.to_csv(csv_path, index=False)
        agent = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
            csv_path,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
    
        respuesta = agent.run(prompt_question)

        os.remove(csv_path)

        print(respuesta)
        return respuesta
    else:
        # La solicitud HTTP no fue exitosa
        print(f'Error al descargar el archivo: {response.status_code}')

def index(request):
    return render(request,"index/index-2.html")
def registro_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        if password1 == password2:
            user = User.objects.create_user(username=username, email=email, password=password1)

            login(request, user)
            return redirect('index')
        else:
            error_message = "Las contraseñas no coinciden."
    else:
        error_message = ""

    return render(request, 'account/signup.html', {'error_message': error_message})

@login_required
def logout_view(request):
    logout(request)
    return redirect('index')


def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            error_message = "Contraseña o usuario inválido."
    else:
        error_message = ""
    
    return render(request, 'account/login.html', {'error_message': error_message})

def download_csv_from_url(url, local_filename):
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(local_filename, 'wb') as csv_file:
            csv_file.write(response.content)


def dashboard_view(request):
    archivo_url = request.session.get('archivo_url')
    print(archivo_url)
    if request.method == 'POST':
        if ('pdf') in archivo_url:  
            print("PDF")    
            prompt_question = request.POST['prompt_question']
            response = get_response_pdf(request, prompt_question, archivo_url)
            content = response.content.decode('utf-8')
            response_data = json.loads(content)
            context = {'prompt_question': prompt_question, 'response_pdf': response_data}
        elif ('csv') in archivo_url:
            print("CSV")
            prompt_question = request.POST['prompt_question']
            response_data = get_response_csv(request, prompt_question, archivo_url)
            context = {'prompt_question': prompt_question, 'response_csv': response_data}
        elif ('xls') in archivo_url:
            print("XLS")
            prompt_question = request.POST['prompt_question']
            response_data = get_response_xlsx(request, prompt_question, archivo_url)
            download_csv_from_url(archivo_url, 'local.csv')
            print("CSV")
            df = pd.read_csv('local.csv', encoding = "ISO-8859-1", on_bad_lines='skip')
            print(df.to_numpy())
            context = {'prompt_question': prompt_question, 'response_excel': response_data}
            print(context)
        return render(request, 'dashboard/dashboard.html', context)
    else:
        return render(request, 'dashboard/dashboard.html')

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

            return redirect('dashboard')

    else:
        form = ArchivoForm()

    return render(request, 'dashboard/upload.html', {'form': form})

