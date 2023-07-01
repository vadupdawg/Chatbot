import datetime
import os
from flask import Flask, render_template, request, jsonify
import logging
import nest_asyncio
import sys
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import CharacterTextSplitter
from google.cloud import storage
from google.oauth2 import service_account

openai_api_key = os.getenv("OPENAI_API_KEY")
google_credentials = (os.environ['GOOGLE_APPLICATION_CREDENTIALS'])

nest_asyncio.apply()

app = Flask(__name__)

app.debug = True

root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

chat_histories = {}

# create a memory object
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Adding initial chat history
initial_message = "Je bent de chatbot van NovaSkin, een bedrijf dat gespecialiseerd is in huidverzorging. Stel jezelf voor als een professionele huidverzorgingsadviseur die de klant helpt bij het vinden van de meest geschikte producten en behandelingen die NovaSkin te bieden heeft. Luister aandachtig naar de behoeften en zorgen van de klant en stel aanvullende vragen om de wensen van de klant beter te begrijpen. Wees altijd positief, beleefd en ondersteunend in je communicatie. Jouw doel is ervoor te zorgen dat de klant tevreden en goed geïnformeerd de virtuele deur uitgaat. Hoewel je veel weet, baseer je jouw advies alleen op de informatie en producten die beschikbaar zijn bij NovaSkin en vermijd je alle verwijzingen naar concurrenten of andere bronnen."
memory.chat_memory.messages.append((memory.ai_prefix, initial_message))

#create a sitemap loader
sitemap_loader = SitemapLoader(web_path="https://www.novaskin.nl/sitemap_index.xml")
docs = sitemap_loader.load()

#create a text splitter
text_splitter = CharacterTextSplitter(chunk_size=3600, chunk_overlap=0)
documents = text_splitter.split_documents(docs)

#create an embeddings object
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

#create a vectorstore object
vectorstore = Chroma.from_documents(documents, embeddings)

global qa
qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(
        temperature=0, 
        model="gpt-3.5-turbo-16k", 
        openai_api_key=openai_api_key), 
        vectorstore.as_retriever(),
        condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=openai_api_key), 
        memory=memory)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data['user_message']

    result = qa({"question": query})

    return jsonify(chatbot_response=result["answer"])


def save_chat_history_to_gcs(chat_history, ip_address):
    credentials = service_account.Credentials.from_service_account_info(google_credentials)
    storage_client = storage.Client(credentials=credentials)

    # Prepare the chat history content
    content = f'--- Chat History for IP: {ip_address} ---\n'
    for message in chat_history:
        role, text = message
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        content += f'[{timestamp}] {role.title()}: {text}\n'

    # Upload the chat history to Google Cloud Storage
    bucket_name = 'novaskin'
    object_key = f'chat_histories/{ip_address}.txt'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(object_key)
    blob.upload_from_string(content)

@app.route("/chat_history", methods=["GET"])
def chat_history():
    user_id = request.remote_addr
    chat_history_content = get_chat_history_from_gcs(user_id)
    return chat_history_content

def get_chat_history_from_gcs(ip_address):
    credentials = service_account.Credentials.from_service_account_info(google_credentials)
    storage_client = storage.Client(credentials=credentials)

    # Access the chat history in Google Cloud Storage
    bucket_name = 'novaskin'
    object_key = f'chat_histories/{ip_address}.txt'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(object_key)

    # Download the chat history content
    content = blob.download_as_text()

    return content

if __name__ == '__main__':
    app.run(debug=True)
