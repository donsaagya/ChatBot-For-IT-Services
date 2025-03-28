# This Code is not mine!
# Source: Patrick Loeber. pytorch-chatbot at master · patrickloeber/pytorch-chatbot. GitHub. Retrieved April 21, 2024 from https://github.com/patrickloeber/pytorch-chatbot/blob/master
#

import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

responses_for_unrecognised_queries = [
    "If you're unsure, you can navigate to 'Home' > 'The University' > 'Information Services and Systems' > 'IT Services' on our site. From there, follow the relevant links to submit a query to the IT support team.",
    "I am afraid I can't help you here. Please visit our website and go to 'Home', then 'The University', followed by 'Information Services and Systems', and 'IT Services'. You’ll find links there to get help from our IT staff.",
    "It seems I won't be able to help you on my own. For more assistance, please go to 'Home', select 'The University', then 'Information Services and Systems', and finally 'IT Services' on our website. Here, you can find the appropriate links to contact IT support.",
    "I'm here to help, but you need more detailed assistance, please access 'Home' > 'The University' > 'Information Services and Systems' > 'IT Services' on our website and follow the necessary links to connect with our IT services team.",
    "I won't be able to solve your issue, you can always raise a query with our IT staff by going to 'Home', then 'The University', 'Information Services and Systems', and 'IT Services' on our website. Follow the links provided there for support.",
    "For further assistance, please direct your browser to 'Home', then 'The University', followed by 'Information Services and Systems' and 'IT Services'. There, you can find links to help you connect with our IT support team.",
    "I may need a bit more to fully assist you. You can try going to 'Home' > 'The University' > 'Information Services and Systems' > 'IT Services' on our site and follow the relevant links to contact IT support.",
    "I don't have the authority to fix that, you might find it helpful to visit 'Home', choose 'The University', then 'Information Services and Systems', and finally 'IT Services' on our website. From there, you can follow the links to ask questions directly to our IT staff.",
    "For more specific assistance, please head over to 'Home', then 'The University', click on 'Information Services and Systems', and 'IT Services'. Here, you can follow the links to get the help you need from our IT support team."
]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "ITS Bot"
print("Hello How Can I help you? (type 'quit' to exit)")
sentence = ""
while True:
    # sentence = "need help with account creation"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:#change to 0.75
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print("--------")
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: {random.choice(responses_for_unrecognised_queries)}")

    print("---------------")
    print("---------------")