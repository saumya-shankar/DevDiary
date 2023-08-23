# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 07:16:05 2023

@author: vijmr
"""
import re
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
model = GPT2LMHeadModel.from_pretrained("./song_generator_model")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def preprocess_text(content):
  # i) Convert to lowercase
  content = content.lower()

  # ii) Remove numbers
  content = re.sub(r'\d+', '', content)

  # iii) Remove first sentence
  sentences = content.split('\n')
  if len(sentences) > 1:
    content = '\n'.join(sentences[1:])

  # iv) Remove 'Embed' in last sentence
  sentences = content.split('\n')
  if len(sentences) > 0:
    last_sentence = sentences[-1]
    sentences[-1] = last_sentence.replace('embed', '')
    content = '\n'.join(sentences)

  # v) Replace \n with "<newline>"
  content = content.replace('\n', ' <newline> ')

  # vi) Remove all extra spaces
  content = re.sub(' +', ' ', content) # replace multiple spaces with a single space
  content = content.strip()  # remove leading and trailing spaces
    
  return content

def generate_song(input_str):
    input_str="start\n"+input_str
    input_str=preprocess_text(input_str)
    inputs = tokenizer.encode(input_str, return_tensors='pt')
    outputs = model.generate(inputs, max_length=1024, temperature=0.7, num_return_sequences=1)
    return tokenizer.decode(outputs[0])

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    input_str = data.get('input_str')
    result = generate_song(input_str)
    return jsonify({'song': result})

@app.route('/test', methods=['GET'])
def test():
    input_str = "My heart is filled with love"
    song = generate_song(input_str)
    return jsonify({'song': song})
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

