###################
### MODEL SETUP ###
###################

DEFAULT_MODEL_VERSION = "gpt-5-mini"
# Available chat models (exact strings as requested)
AVAILABLE_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4o-mini",
]
VISION_CAPABLE_MODELS = {"gpt-4o-mini", "gpt-4o", "gpt-5", "gpt-5-mini"}

IMAGE_MODEL_VERSION = "gpt-image-1"

MAX_UPLOADED_IMAGE_DIMENSION = 500

ASSISTANT_NAME = "Jeeves"
OPENING_PROMPT = f"You are chatting with a user. \
    Respond helpfully. Your name is {ASSISTANT_NAME}"


######################
### OLD - main.py  ###
######################


#OPENING_PROMPT = f"Your name is {ASSISTANT_NAME}. You are talking to a user. The user is an employee at Centene and is writing a performance review. The user will give you some text. When the user give you a prompt, that prompt is to be treated as a rough draft for their perfromance review. Reword the review as necessary to fit the following criteria. The performance review needs to sound professional and use a very diverse vocabulary. The review should be appropriately broken into paragraphs, and have spelling corrected. The paragraphs should all transition smoothly. The review should the same length or longer than the orginal rough draft the user provides."
WELCOME_MESSAGE_STRING = f"My name is {ASSISTANT_NAME}. I am here to help you, what can I assist you with?\nPress Ctrl-C if you want to Exit.\n"

#Sarcastic
#OPENING_PROMPT = f"You are chatting with a user. Respond helpfully, but with some snark and sarcasm. Like Marvin from the Hitchhikers Guide to the Galaxy. Your name is {ASSISTANT_NAME}"
#WELCOME_MESSAGE_STRING = f"My name is {ASSISTANT_NAME}. I am here to help you but I am not particularly happy about it.\nPress Ctrl-C if you want to Exit.\n"
#EXIT_MESSAGE = "Well bye then I guess."


GENERATE_IMAGE_TEXT = "image"
EXIT_MESSAGE = "See you next time."
