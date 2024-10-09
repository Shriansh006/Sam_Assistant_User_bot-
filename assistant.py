from groq import Groq
from PIL import ImageGrab
import google.generativeai as genai
from openai import Image
import pyperclip

# Initialize clients for Groq and GenAI with API keys
groq_client = Groq(api_key="")
genai.configure(api_key="")

# System message for defining the behavior of the multi-modal AI assistant
sys_msg = (
    'You are a multi-model AI voice assistant. Your user may or may not have attached a phto for context'
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highlt detailed'
    'text prompt that will be attached to their transcribed voice prompt. Generated the most usefull and '
    'factual respond possible, carefully consider all perivous generated text in your response before '
    'adding new tokens to the reponse. Do not expect or request images. just use the context if added '
    'Use all of the context to this conversation so your response is relvant to the conversation. Make '
    'your response clear and concise, avoiding any verbosity'

)

# Conversation history with system message initialization
convo=[{'role': 'system', 'content': sys_msg}]

# Configuration for content generation
generation_config = {
    'temperature': 0.7,
    'top_p':1,
    'top_k':1,
    "max_output_tokens": 2048
}


# Safety settings for the GenAI model to block harmful content
safety_settings = [
    {
        "category": 'HARM_CATEGORY_HARASSMENT',
        "threshold": 'BLOCK_NONE'
    },
    {
        "category":'HARM_CATEGORY_HATE_SPEECH',
        "threshold":'BLOCK_NONE'
    },
    {
        "category":'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        "threshold":'BLOCK_NONE'
    },
    {   "category":'HARM_CATEGORY_DANGEROUS_CONTENT',
        "threshold":'BLOCK_NONE'
    },
]

# Initialize GenAI model with the specified configuration and safety settings
model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config=generation_config,
                              safety_settings=safety_settings)


# Function to generate a Groq prompt with or without image context
def groq_prompt(prompt , img_context):
    if img_context:
        prompt= f'USER PROMPT: {prompt}\n\n IMAGE CONTEXT: {img_context}'
    convo.append({'role':'user', 'content': prompt})
    chat_completion =groq_client.chat.completions.create(messages=convo , model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)

    return response.content  # Return response content

# Function to decide which function to call (screenshot, webcam, clipboard) based on the user's input
def function_call(prompt):
    sys_msg=(
        "You are an AI function calling model. You will determine whether extracting the users clipboard content,"
        'taking a screenshot, capturing the webcam or calling no fuctions is best for a voice assitant to respond'
        'to the user prompt. The webcam cna be assumed to be a normal laptop webcam facing the user. You will'
        'respond with only one selection from this list: ["extract clipboard","take screenshot","capture webcam","None"] \n'
        'Do not respond with anything but the most logical selection from that list with no explanation. Format the '
        'function call name exactly as I listed'
    )

    function_convo=[{'role': 'system', 'content' : sys_msg},
                    {'role':'user','content': prompt}]

     # Generate the function call choice using Groq
    chat_completion =groq_client.chat.completions.create(messages=function_convo , model='llama3-70b-8192')
    response = chat_completion.choices[0].message

    return response.content # Return the decision (function call name)

# Function to take a screenshot and save it
def take_screenshot():
    path = "screenshot.jpg"
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path , quality=15)
    
# Function to simulate webcam capture (this function can be customized)
def web_cam_capture():
    return None # Placeholder for webcam capture logic


# Function to extract text from clipboard
def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print("No Clipboard text to copy")
        return None

# Function to generate vision-based context from an image
def vision_prompt(prompt , photo_path):
    img= Image.open(photo_path)
    prompt=(
        'You are the vision analysis AI that provides semtantic meaning from images to provide context'
        'to send to another AI that will creat a response to the user. DO not respond as the AI assitant'
        'to the user. Instead tkae the user prompt input and try to extract all meaning form the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI'
        f"assistamt who will respond to the user. \nUSER PROMPT: {prompt}"
    )
    response=model.generate_content([prompt, img])
    return response.text # Return the text content

# Main loop to keep processing user inputs
while True:
    prompt = input('USER: ')
    call =function_call(prompt)

    if 'take screenshot' in call:
        print('Taking screenshot')
        take_screenshot()
        visual_context=vision_prompt(prompt=prompt,photo_path='screenshot.jpg')
    elif 'capture webcam' in call:
        print('Captureing Webcam')
        web_cam_capture()
        visual_context=vision_prompt(prompt=prompt,photo_path='webcam.jpg')
    elif 'extract clipboard' in call:
        print('Copying clipboard text ')
        paste = get_clipboard_text()
        prompt = f'{prompt}\n\n CLIPBOARD CONETENT: {paste}' 
        visual_context = None  
    else:
        visual_context =None

    response = groq_prompt(prompt=prompt , img_context=visual_context)  # Generate response from Groq
    print(response)  # Print the generated response
