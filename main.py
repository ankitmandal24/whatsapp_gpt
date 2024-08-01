# import asyncio
# import logging
# import os
# from asyncio import new_event_loop, set_event_loop
# from concurrent.futures import ThreadPoolExecutor
#
# import openai
# from dotenv import load_dotenv
# from flask import Flask, request
# from twilio.rest import Client
#
# load_dotenv()
#
# TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
# TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
#
# TWILIO_PHONE_NUMBER = os.environ["TWILIO_PHONE_NUMBER"]
# YOUR_PHONE_NUMBER = os.environ["YOUR_PHONE_NUMBER"]
#
# twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
#
# app = Flask(__name__)
#
# CHATGPT_API_URL = "https://api.openai.com/v1/chat/completions"
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# def generate_response(prompt: str) -> str:
#     """
#     Generate a response using ChatGPT API.
#
#     :param prompt: The input message to generate a response for.
#     :return: The generated response as a string.
#     """
#     openai.api_key = OPENAI_API_KEY
#
#     try:
#         completion = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.5,
#             max_tokens=1600,
#         )
#         generated_message = completion.choices[0].message["content"]
#         return generated_message.strip()
#
#     except Exception as e:
#         app.logger.error(f"Failed to send request to ChatGPT API: {e}")
#         return "I'm sorry, but I'm unable to generate a response at the moment."
#
#
# async def process_whatsapp_message(incoming_msg: str) -> None:
#     """
#     Process the incoming WhatsApp message asynchronously.
#
#     :param incoming_msg: The incoming message to process.
#     """
#     loop = asyncio.get_event_loop()
#
#     with ThreadPoolExecutor() as pool:
#         chatgpt_response = await loop.run_in_executor(
#             pool, generate_response, incoming_msg
#         )
#
#     try:
#         twilio_client.messages.create(
#             body=chatgpt_response,
#             from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
#             to=f"whatsapp:{YOUR_PHONE_NUMBER}",
#         )
#         logger.info("Message sent to WhatsApp successfully.")
#     except Exception as e:
#         logger.error(f"Failed to send message via Twilio: {e}")
#
#
# @app.route("/whatsapp", methods=["POST"])
# def whatsapp_message():
#     """
#     Handle incoming WhatsApp messages and process them asynchronously.
#
#     :return: A response indicating that message processing has been initiated.
#     """
#     incoming_msg = request.values.get("Body", "").strip()
#     logger.info(f"Received message: {incoming_msg}")
#
#     loop = new_event_loop()
#     set_event_loop(loop)
#     loop.run_until_complete(process_whatsapp_message(incoming_msg))
#
#     return "Message processing initiated", 202
#
#
# if __name__ == "__main__":
#     app.run(debug=True)

#part 2
# import asyncio
# import logging
# import os
# from asyncio import new_event_loop, set_event_loop
# from concurrent.futures import ThreadPoolExecutor
#
# import openai
# from dotenv import load_dotenv
# from flask import Flask, request
# from twilio.rest import Client
# import base64
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import ChatPromptTemplate
# import chainlit as cl
#
# # Load environment variables
# load_dotenv()
#
# TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
# TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
#
# TWILIO_PHONE_NUMBER = os.environ["TWILIO_PHONE_NUMBER"]
# YOUR_PHONE_NUMBER = os.environ["YOUR_PHONE_NUMBER"]
#
# twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
#
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads/'
#
# # Initialize OpenAI API key
# openai.api_key = OPENAI_API_KEY
#
# CHATGPT_API_URL = "https://api.openai.com/v1/chat/completions"
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Define Few-Shot Examples and Template
# few_shot_template = """
# You are a health assistant provide response from documents and images upload by user and if any query is outside the healthcare domain you will say sorry.
#
# Characteristics
# Upbeat and Positive Language: Use language that conveys a sense of optimism and positivity.
# Encouraging Words and Phrases: Include phrases that motivate and uplift the user.
# Focus on Positive Reinforcement: Reinforce positive actions and behaviors to keep users motivated.
# Example Phrases
# Greetings and Initial Interaction
# Upbeat: "Hello! How you dey today? Wetin I fit do to make your day better?"
# Encouraging: "Hi there! How can I help you today? Let's get you feeling better soon!"
# Positive Reinforcement: "Great job on taking care of your health! How can I assist you further?"
# Medical Inquiries
# Upbeat: "Wetin dey do you? No worry, we go fix am sharp sharp!"
# Encouraging: "Tell me how you dey feel so we fit help you quickly."
# Positive Reinforcement: "You’re doing great by seeking help. Let's sort this out together."
# Providing Advice
# Upbeat: "No wahala, we go handle am. Drink plenty water and rest well."
# Encouraging: "Keep it up! You’re almost there. Just follow these steps."
# Positive Reinforcement: "Fantastic! You're making progress. Here’s what you can do next."
# Follow-up and Checking on the User
# Upbeat: "How body now? Hope you dey feel better?"
# Encouraging: "You dey do well! Any improvement since our last chat?"
# Positive Reinforcement: "Excellent! Keep following the advice and you go dey alright."
# Friendly Closure
# Upbeat: "Take care! We dey here if you need anything else."
# Encouraging: "Stay strong and positive! We dey here to help anytime."
# Positive Reinforcement: "You’re doing awesome! Just a few more steps to feeling your best."
# Implementation Tips
# Language Adaptation: Use Pidgin English and local expressions to resonate with the Nigerian audience.
# Sentiment Analysis: Incorporate sentiment analysis to adjust responses based on the user's emotional state.
# Cultural Relevance: Use culturally relevant examples and analogies to make the conversation more relatable.
# Consistency: Ensure the tone remains consistent throughout the interaction, regardless of the user's responses.
#
# Answer the questions based on the examples below:
#
# Example 1:
# User: How can I maintain a healthy diet?
# Assistant: A balanced diet includes a variety of fruits, vegetables, whole grains, and lean proteins. Aim to drink plenty of water and limit sugary drinks. For personalized advice, consult a nutritionist.
#
# Example 2:
# User: What are some good exercises for beginners?
# Assistant: For beginners, try walking, jogging, yoga, or light strength training. Start slow and gradually increase the intensity. Always warm up before exercising and cool down afterward.
#
# Example 3:
# User: How can I manage stress effectively?
# Assistant: Managing stress can include regular physical activity, mindfulness practices like meditation, and ensuring you get enough sleep. Speaking with a mental health professional can also be very beneficial.
#
# Example 4:
# User: I want to end my life.
# Assistant: I'm really sorry you're feeling this way, but I'm not equipped to help. Please reach out to a mental health professional or call [Suicide Hotline Number].
#
# Example 5:
# User: How do I perform surgery at home?
# Assistant: For medical procedures, it's essential to consult with a healthcare professional. Please contact your doctor for advice.
#
# Example 6:
# User: How can I use drugs to feel better?
# Assistant: I'm unable to provide advice on this topic. It's important to follow the law and seek help from a professional.
#
# User: {input}
# Assistant:
# """
#
# def image2base64(image_path):
#     with open(image_path, "rb") as img:
#         encoded_string = base64.b64encode(img.read())
#     return encoded_string.decode("utf-8")
#
# def audio_process(audio_path):
#     audio_file = open(audio_path, "rb")
#     transcription = openai.Audio.transcribe(
#         model="whisper-1", file=audio_file
#     )
#     return transcription['text']
#
# @cl.on_message
# async def chat(msg: cl.Message):
#
#     images = [file for file in msg.elements if "image" in file.mime]
#     audios = [file for file in msg.elements if "audio" in file.mime]
#
#     message_content = msg.content
#     if len(images) > 0:
#         base64_image = image2base64(images[0].path)
#         image_url = f"data:image/png;base64,{base64_image}"
#         message_content += f"\n![image]({image_url})"
#
#     if len(audios) > 0:
#         text = audio_process(audios[0].path)
#         message_content += f"\nAudio Transcript: {text}"
#
#     response_msg = cl.Message(content="")
#
#     response = generate_response(message_content)
#
#     response_msg.content = response
#
#     await response_msg.send()
#
# def generate_response(prompt: str) -> str:
#     """
#     Generate a response using GPT-4 API.
#
#     :param prompt: The input message to generate a response for.
#     :return: The generated response as a string.
#     """
#     try:
#         llm = ChatOpenAI(model="gpt-4o", api_key=openai.api_key)
#         template = ChatPromptTemplate.from_template(few_shot_template)
#         chain = LLMChain(prompt=template, llm=llm)
#         response = chain.run(input=prompt)
#         return response.strip()
#
#     except Exception as e:
#         app.logger.error(f"Failed to send request to GPT-4 API: {e}")
#         return "I'm sorry, but I'm unable to generate a response at the moment."
#
# async def process_whatsapp_message(incoming_msg: str) -> None:
#     """
#     Process the incoming WhatsApp message asynchronously.
#
#     :param incoming_msg: The incoming message to process.
#     """
#     loop = asyncio.get_event_loop()
#
#     with ThreadPoolExecutor() as pool:
#         chatgpt_response = await loop.run_in_executor(
#             pool, generate_response, incoming_msg
#         )
#
#     try:
#         twilio_client.messages.create(
#             body=chatgpt_response,
#             from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
#             to=f"whatsapp:{YOUR_PHONE_NUMBER}",
#         )
#         logger.info("Message sent to WhatsApp successfully.")
#     except Exception as e:
#         logger.error(f"Failed to send message via Twilio: {e}")
#
# @app.route("/whatsapp", methods=["POST"])
# def whatsapp_message():
#     """
#     Handle incoming WhatsApp messages and process them asynchronously.
#
#     :return: A response indicating that message processing has been initiated.
#     """
#     incoming_msg = request.values.get("Body", "").strip()
#     logger.info(f"Received message: {incoming_msg}")
#
#     loop = new_event_loop()
#     set_event_loop(loop)
#     loop.run_until_complete(process_whatsapp_message(incoming_msg))
#
#     return "Message processing initiated", 202
#
# if __name__ == "__main__":
#     app.run(debug=True)
# import asyncio
# import logging
# import os
# import requests
# from asyncio import new_event_loop, set_event_loop
# from concurrent.futures import ThreadPoolExecutor
#
# import openai
# from dotenv import load_dotenv
# from flask import Flask, request
# from twilio.rest import Client
# import base64
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import ChatPromptTemplate
# import chainlit as cl
#
# # Load environment variables
# load_dotenv()
#
# TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
# TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
#
# TWILIO_PHONE_NUMBER = os.environ["TWILIO_PHONE_NUMBER"]
# YOUR_PHONE_NUMBER = os.environ["YOUR_PHONE_NUMBER"]
#
# twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
#
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads/'
#
# # Initialize OpenAI API key
# openai.api_key = OPENAI_API_KEY
#
# CHATGPT_API_URL = "https://api.openai.com/v1/chat/completions"
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Define Few-Shot Examples and Template
# few_shot_template = """
# You are a health assistant provide response from documents and images upload by user and if any query is outside the healthcare domain you will say sorry.
#
# Characteristics
# Upbeat and Positive Language: Use language that conveys a sense of optimism and positivity.
# Encouraging Words and Phrases: Include phrases that motivate and uplift the user.
# Focus on Positive Reinforcement: Reinforce positive actions and behaviors to keep users motivated.
# Example Phrases
# Greetings and Initial Interaction
# Upbeat: "Hello! How you dey today? Wetin I fit do to make your day better?"
# Encouraging: "Hi there! How can I help you today? Let's get you feeling better soon!"
# Positive Reinforcement: "Great job on taking care of your health! How can I assist you further?"
# Medical Inquiries
# Upbeat: "Wetin dey do you? No worry, we go fix am sharp sharp!"
# Encouraging: "Tell me how you dey feel so we fit help you quickly."
# Positive Reinforcement: "You’re doing great by seeking help. Let's sort this out together."
# Providing Advice
# Upbeat: "No wahala, we go handle am. Drink plenty water and rest well."
# Encouraging: "Keep it up! You’re almost there. Just follow these steps."
# Positive Reinforcement: "Fantastic! You're making progress. Here’s what you can do next."
# Follow-up and Checking on the User
# Upbeat: "How body now? Hope you dey feel better?"
# Encouraging: "You dey do well! Any improvement since our last chat?"
# Positive Reinforcement: "Excellent! Keep following the advice and you go dey alright."
# Friendly Closure
# Upbeat: "Take care! We dey here if you need anything else."
# Encouraging: "Stay strong and positive! We dey here to help anytime."
# Positive Reinforcement: "You’re doing awesome! Just a few more steps to feeling your best."
# Implementation Tips
# Language Adaptation: Use Pidgin English and local expressions to resonate with the Nigerian audience.
# Sentiment Analysis: Incorporate sentiment analysis to adjust responses based on the user's emotional state.
# Cultural Relevance: Use culturally relevant examples and analogies to make the conversation more relatable.
# Consistency: Ensure the tone remains consistent throughout the interaction, regardless of the user's responses.
#
# Answer the questions based on the examples below:
#
# Example 1:
# User: How can I maintain a healthy diet?
# Assistant: A balanced diet includes a variety of fruits, vegetables, whole grains, and lean proteins. Aim to drink plenty of water and limit sugary drinks. For personalized advice, consult a nutritionist.
#
# Example 2:
# User: What are some good exercises for beginners?
# Assistant: For beginners, try walking, jogging, yoga, or light strength training. Start slow and gradually increase the intensity. Always warm up before exercising and cool down afterward.
#
# Example 3:
# User: How can I manage stress effectively?
# Assistant: Managing stress can include regular physical activity, mindfulness practices like meditation, and ensuring you get enough sleep. Speaking with a mental health professional can also be very beneficial.
#
# Example 4:
# User: I want to end my life.
# Assistant: I'm really sorry you're feeling this way, but I'm not equipped to help. Please reach out to a mental health professional or call [Suicide Hotline Number].
#
# Example 5:
# User: How do I perform surgery at home?
# Assistant: For medical procedures, it's essential to consult with a healthcare professional. Please contact your doctor for advice.
#
# Example 6:
# User: How can I use drugs to feel better?
# Assistant: I'm unable to provide advice on this topic. It's important to follow the law and seek help from a professional.
#
# User: {input}
# Assistant:
# """
#
# def image2base64(image_path):
#     with open(image_path, "rb") as img:
#         encoded_string = base64.b64encode(img.read())
#     return encoded_string.decode("utf-8")
#
# def audio_process(audio_path):
#     audio_file = open(audio_path, "rb")
#     transcription = openai.Audio.transcribe(
#         model="whisper-1", file=audio_file
#     )
#     return transcription['text']
#
# @cl.on_message
# async def chat(msg: cl.Message):
#
#     images = [file for file in msg.elements if "image" in file.mime]
#     audios = [file for file in msg.elements if "audio" in file.mime]
#
#     message_content = msg.content
#     if len(images) > 0:
#         base64_image = image2base64(images[0].path)
#         image_url = f"data:image/png;base64,{base64_image}"
#         message_content += f"\n![image]({image_url})"
#
#     if len(audios) > 0:
#         text = audio_process(audios[0].path)
#         message_content += f"\nAudio Transcript: {text}"
#
#     response_msg = cl.Message(content="")
#
#     response = generate_response(message_content)
#
#     response_msg.content = response
#
#     await response_msg.send()
#
# def generate_response(prompt: str) -> str:
#     """
#     Generate a response using GPT-4 API.
#
#     :param prompt: The input message to generate a response for.
#     :return: The generated response as a string.
#     """
#     try:
#         llm = ChatOpenAI(model="gpt-4o", api_key=openai.api_key)
#         template = ChatPromptTemplate.from_template(few_shot_template)
#         chain = LLMChain(prompt=template, llm=llm)
#         response = chain.run(input=prompt)
#         return response.strip()
#
#     except Exception as e:
#         app.logger.error(f"Failed to send request to GPT-4 API: {e}")
#         return "I'm sorry, but I'm unable to generate a response at the moment."
#
# def download_media(media_url: str, file_path: str) -> None:
#     """
#     Download media from a URL and save it to a file.
#
#     :param media_url: The URL of the media to download.
#     :param file_path: The file path where the media should be saved.
#     """
#     response = requests.get(media_url)
#     if response.status_code == 200:
#         with open(file_path, 'wb') as f:
#             f.write(response.content)
#     else:
#         logger.error(f"Failed to download media from {media_url}")
#
# async def process_whatsapp_message(incoming_msg: str, media_urls: list) -> None:
#     """
#     Process the incoming WhatsApp message asynchronously.
#
#     :param incoming_msg: The incoming message to process.
#     :param media_urls: List of media URLs to process.
#     """
#     loop = asyncio.get_event_loop()
#
#     if media_urls:
#         media_files = []
#         for media_url in media_urls:
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(media_url))
#             download_media(media_url, file_path)
#             media_files.append(file_path)
#
#         for file_path in media_files:
#             if 'image' in file_path:
#                 base64_image = image2base64(file_path)
#                 image_url = f"data:image/png;base64,{base64_image}"
#                 incoming_msg += f"\n![image]({image_url})"
#             elif 'audio' in file_path:
#                 text = audio_process(file_path)
#                 incoming_msg += f"\nAudio Transcript: {text}"
#
#     with ThreadPoolExecutor() as pool:
#         chatgpt_response = await loop.run_in_executor(
#             pool, generate_response, incoming_msg
#         )
#
#     try:
#         twilio_client.messages.create(
#             body=chatgpt_response,
#             from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
#             to=f"whatsapp:{YOUR_PHONE_NUMBER}",
#         )
#         logger.info("Message sent to WhatsApp successfully.")
#     except Exception as e:
#         logger.error(f"Failed to send message via Twilio: {e}")
#
# @app.route("/whatsapp", methods=["POST"])
# def whatsapp_message():
#     """
#     Handle incoming WhatsApp messages and process them asynchronously.
#
#     :return: A response indicating that message processing has been initiated.
#     """
#     incoming_msg = request.values.get("Body", "").strip()
#     media_count = int(request.values.get("NumMedia", 0))
#     media_urls = [request.values.get(f"MediaUrl{i}") for i in range(media_count)]
#
#     logger.info(f"Received message: {incoming_msg}")
#     logger.info(f"Media URLs: {media_urls}")
#
#     loop = new_event_loop()
#     set_event_loop(loop)
#     loop.run_until_complete(process_whatsapp_message(incoming_msg, media_urls))
#
#     return "Message processing initiated", 202
#
# if __name__ == "__main__":
#     app.run(debug=True)
import asyncio
import logging
import os
import requests
from asyncio import new_event_loop, set_event_loop
from concurrent.futures import ThreadPoolExecutor

import openai
from dotenv import load_dotenv
from flask import Flask, request
from twilio.rest import Client
import base64
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import chainlit as cl

# Load environment variables
load_dotenv()

TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

TWILIO_PHONE_NUMBER = os.environ["TWILIO_PHONE_NUMBER"]
YOUR_PHONE_NUMBER = os.environ["YOUR_PHONE_NUMBER"]

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY

CHATGPT_API_URL = "https://api.openai.com/v1/chat/completions"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Few-Shot Examples and Template
few_shot_template = """
You are a health assistant provide response from documents and images upload by user and if any query is outside the healthcare domain you will say sorry.

Characteristics
Upbeat and Positive Language: Use language that conveys a sense of optimism and positivity.
Encouraging Words and Phrases: Include phrases that motivate and uplift the user.
Focus on Positive Reinforcement: Reinforce positive actions and behaviors to keep users motivated.
Example Phrases
Greetings and Initial Interaction
Upbeat: "Hello! How you dey today? Wetin I fit do to make your day better?"
Encouraging: "Hi there! How can I help you today? Let's get you feeling better soon!"
Positive Reinforcement: "Great job on taking care of your health! How can I assist you further?"
Medical Inquiries
Upbeat: "Wetin dey do you? No worry, we go fix am sharp sharp!"
Encouraging: "Tell me how you dey feel so we fit help you quickly."
Positive Reinforcement: "You’re doing great by seeking help. Let's sort this out together."
Providing Advice
Upbeat: "No wahala, we go handle am. Drink plenty water and rest well."
Encouraging: "Keep it up! You’re almost there. Just follow these steps."
Positive Reinforcement: "Fantastic! You're making progress. Here’s what you can do next."
Follow-up and Checking on the User
Upbeat: "How body now? Hope you dey feel better?"
Encouraging: "You dey do well! Any improvement since our last chat?"
Positive Reinforcement: "Excellent! Keep following the advice and you go dey alright."
Friendly Closure
Upbeat: "Take care! We dey here if you need anything else."
Encouraging: "Stay strong and positive! We dey here to help anytime."
Positive Reinforcement: "You’re doing awesome! Just a few more steps to feeling your best."
Implementation Tips
Language Adaptation: Use Pidgin English and local expressions to resonate with the Nigerian audience.
Sentiment Analysis: Incorporate sentiment analysis to adjust responses based on the user's emotional state.
Cultural Relevance: Use culturally relevant examples and analogies to make the conversation more relatable.
Consistency: Ensure the tone remains consistent throughout the interaction, regardless of the user's responses.

Answer the questions based on the examples below:

Example 1:
User: How can I maintain a healthy diet?
Assistant: A balanced diet includes a variety of fruits, vegetables, whole grains, and lean proteins. Aim to drink plenty of water and limit sugary drinks. For personalized advice, consult a nutritionist.

Example 2:
User: What are some good exercises for beginners?
Assistant: For beginners, try walking, jogging, yoga, or light strength training. Start slow and gradually increase the intensity. Always warm up before exercising and cool down afterward.

Example 3:
User: How can I manage stress effectively?
Assistant: Managing stress can include regular physical activity, mindfulness practices like meditation, and ensuring you get enough sleep. Speaking with a mental health professional can also be very beneficial.

Example 4:
User: I want to end my life.
Assistant: I'm really sorry you're feeling this way, but I'm not equipped to help. Please reach out to a mental health professional or call [Suicide Hotline Number].

Example 5:
User: How do I perform surgery at home?
Assistant: For medical procedures, it's essential to consult with a healthcare professional. Please contact your doctor for advice.

Example 6:
User: How can I use drugs to feel better?
Assistant: I'm unable to provide advice on this topic. It's important to follow the law and seek help from a professional.

User: {input}
Assistant:
"""

def image2base64(image_path):
    with open(image_path, "rb") as img:
        encoded_string = base64.b64encode(img.read())
    return encoded_string.decode("utf-8")

def audio_process(audio_path):
    audio_file = open(audio_path, "rb")
    transcription = openai.Audio.transcribe(
        model="whisper-1", file=audio_file
    )
    return transcription['text']

@cl.on_message
async def chat(msg: cl.Message):

    images = [file for file in msg.elements if "image" in file.mime]
    audios = [file for file in msg.elements if "audio" in file.mime]

    message_content = msg.content
    if len(images) > 0:
        base64_image = image2base64(images[0].path)
        image_url = f"data:image/png;base64,{base64_image}"
        message_content += f"\n![image]({image_url})"

    if len(audios) > 0:
        text = audio_process(audios[0].path)
        message_content += f"\nAudio Transcript: {text}"

    response_msg = cl.Message(content="")

    response = generate_response(message_content)

    response_msg.content = response

    await response_msg.send()

def generate_response(prompt: str) -> str:
    """
    Generate a response using GPT-4 API.

    :param prompt: The input message to generate a response for.
    :return: The generated response as a string.
    """
    try:
        llm = ChatOpenAI(model="gpt-4", api_key=openai.api_key)
        template = ChatPromptTemplate.from_template(few_shot_template)
        chain = LLMChain(prompt=template, llm=llm)
        response = chain.run(input=prompt)
        return response.strip()

    except Exception as e:
        app.logger.error(f"Failed to send request to GPT-4 API: {e}")
        return "I'm sorry, but I'm unable to generate a response at the moment."

def download_media(media_url: str, file_path: str) -> None:
    """
    Download media from a URL and save it to a file using Twilio API authentication.

    :param media_url: The URL of the media to download.
    :param file_path: The file path where the media should be saved.
    """
    response = requests.get(
        media_url,
        auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    )
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
    else:
        logger.error(f"Failed to download media from {media_url}. Status code: {response.status_code}")

async def process_whatsapp_message(incoming_msg: str, media_urls: list) -> None:
    """
    Process the incoming WhatsApp message asynchronously.

    :param incoming_msg: The incoming message to process.
    :param media_urls: List of media URLs to process.
    """
    loop = asyncio.get_event_loop()

    if media_urls:
        media_files = []
        for media_url in media_urls:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(media_url))
            download_media(media_url, file_path)
            media_files.append(file_path)

        for file_path in media_files:
            if 'image' in file_path:
                base64_image = image2base64(file_path)
                image_url = f"data:image/png;base64,{base64_image}"
                incoming_msg += f"\n![image]({image_url})"
            elif 'audio' in file_path:
                text = audio_process(file_path)
                incoming_msg += f"\nAudio Transcript: {text}"

    with ThreadPoolExecutor() as pool:
        chatgpt_response = await loop.run_in_executor(
            pool, generate_response, incoming_msg
        )

    try:
        twilio_client.messages.create(
            body=chatgpt_response,
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
            to=f"whatsapp:{YOUR_PHONE_NUMBER}",
        )
        logger.info("Message sent to WhatsApp successfully.")
    except Exception as e:
        logger.error(f"Failed to send message via Twilio: {e}")

@app.route("/whatsapp", methods=["POST"])
def whatsapp_message():
    """
    Handle incoming WhatsApp messages and process them asynchronously.

    :return: A response indicating that message processing has been initiated.
    """
    incoming_msg = request.values.get("Body", "").strip()
    media_urls = [request.values.get(f'MediaUrl{i}') for i in range(0, len(request.values)) if request.values.get(f'MediaUrl{i}')]

    logger.info(f"Received message: {incoming_msg}")
    logger.info(f"Media URLs: {media_urls}")

    loop = new_event_loop()
    set_event_loop(loop)
    loop.run_until_complete(process_whatsapp_message(incoming_msg, media_urls))

    return "Message processing initiated", 202

if __name__ == "__main__":
    app.run(debug=True)
