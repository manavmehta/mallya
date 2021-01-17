# -*- coding: utf-8 -*-

import sys
import signal
import random
import csv
import requests as requests
from config import TOKEN, BASE_TELEGRAM_URL
from commands import commands
import numpy as np
import gspread
import tensorflow as tf
import tensorflow_hub as hub
from pymongo import MongoClient
from timeloop import Timeloop
from datetime import timedelta
import urllib.parse as url
import pandas as pd

tl = Timeloop()

def getChatID(update):
    '''
        Returns the chatID from the update.
    '''
    return update.message.chat.id


def getUserID(update):
    '''
        Returns the userID from the update.
    '''
    return update.message.from_user.id


def getMessageText(update):
    '''
        Returns the actual text(payload) from the message/edited_message.
        These two are types of updates sent by teegram API.
        message is just a simple message sent by the user.
        edited_message is when the previous message was edited.
    '''
    if (update.message):
        return update.message.text
    else:
        return update.edited_message.text


# create function that get getLastUpdate
def getLastUpdate(req, offset=None):
    '''
        Returns the latest update from the getUpdates bot API call.
    '''
    updates_url=req + 'getUpdates'
    if(offset != None):
        updates_url+='?offset={}'.format(offset)
    response = requests.get(updates_url).json()
    result = response['result']
    return result[-5:]  # get last record message update


def sendMessage(chat_id, message_text):
    '''
        Sends the <message_text> to the corresponding <chat_id>
    '''
    params = {'chat_id': chat_id, 'text': message_text}
    response = requests.post(BASE_TELEGRAM_URL + 'sendMessage', data=params)
    return response


# Bifurcate incoming_message_command into command and incoming message
def bifurcate(incoming_message_command):
    '''
        Bifurcates the command and the message into two
        examples:
        /hi -> hi, None
        /q How is IIT Mandi? -> q, How is IIT Mandi
    '''
    command, incoming_message = None, None
    if (incoming_message_command[0]!='/'): # Any message that does not start with a / is not a valid command
        return 'general', incoming_message_command             # decision tree
    else:
        i=1
        command = ''
        while (i<len(incoming_message_command) and incoming_message_command[i]!=' '):
            command+=incoming_message_command[i]
            i+=1
        if(i<len(incoming_message_command)):
            incoming_message = incoming_message_command[i+1:]
    
    if command in {'u', 'd', 'n', 'q'} or command in commands:
        return command, incoming_message
    else:
        return 'invalid', None


def replyToCommand(chatID, command, first_name):
    '''
        This is called when the message is any valid command except for a query.
        ie. For all commands available in commands.py
    '''
    sendMessage(chatID, commands[command].format(first_name))
    return


def voteOnAnswer(chatID, command, update):                                  # upvotes or downvotes on answers

    global userid_answers_dict, collection_answers

    answer_obj = userid_answers_dict[getUserID(update)][-1]
    
    myquery = { '_id': answer_obj['_id'] }

    state = 0

    if getUserID(update) in answer_obj['upvotes']:
        state = 1
        answer_obj['score'] -= 1
        answer_obj['upvotes'].remove(getUserID(update))
    elif getUserID(update) in answer_obj['downvotes']:
        state = -1
        answer_obj['score'] += 1
        answer_obj['downvotes'].remove(getUserID(update))

    if command == 'u' and state != 1:
        answer_obj['score'] += 1
        answer_obj['upvotes'].append(getUserID(update))
    elif command == 'd' and state != -1:
        answer_obj['score'] -= 1
        answer_obj['downvotes'].append(getUserID(update))
    
    userid_answers_dict[getUserID(update)][-1] = answer_obj
    newvalues = { '$set': { 'upvotes': answer_obj['upvotes'], 'downvotes': answer_obj['downvotes'], 'score': answer_obj['score'] } }
    collection_answers.update_one(myquery, newvalues)

    sendMessage(chatID, "Your Vote was registered !")


def giveOneAnswer(chatID, answers_obj_list):

    answer_obj = answers_obj_list[0]
    for i in range(0, len(answers_obj_list) - 1):
        answers_obj_list[i] = answers_obj_list[i + 1]
    
    answers_obj_list[-1] = answer_obj

    # Above code swap first and last element of the array to cycle through the answers.

    ret_str = answer_obj['text'] + '\n\n' + '___________________________________\n'
    ret_str += '\nUpvotes: {}'.format(answer_obj['score']) + '\n\n' + 'Type /u or /d for upvoting or downvoting this answer or /n for next answer.'

    sendMessage(chatID, ret_str)

    return answers_obj_list


def answerQuery(incoming_message, update):

    global model, questions_text_list, userid_answers_dict, unanswered_questions

    msg_embeddings = model([incoming_message] + questions_text_list)
    answers_obj_list = findAnswer(msg_embeddings)

    # Above code updates answers_obj_list through the NLP model.
    
    if len(answers_obj_list) == 0:
        unanswered_questions.append(incoming_message)
        message="This Question hasn't yet been answered. I will ask my creators to answer it.\nTry a Google search till then:\n"
        search_url='https://www.google.com/search?q={}'.format(url.quote(incoming_message))
        sendMessage(getChatID(update), message+search_url)
    else:
        answers_obj_list = giveOneAnswer(getChatID(update), answers_obj_list)
    
    userid_answers_dict[getUserID(update)] = answers_obj_list               # storing answers list in dict for commands like (/n, /u, /d)

    # if incoming_message in ['hi', 'hello']:
    #     return commands[incoming_message].format(update['message']['from']['first_name'])
    # else:
    #     return commands[incoming_message].format('')

def parseIncomingMessage(update):

    '''
        Returns the chatID from the update.
    '''

    global userid_answers_dict

    incoming_message_command = getMessageText(update).lower()
    print(type(incoming_message_command))
    command, incoming_message = bifurcate(incoming_message_command)
    print('command = ', command, '\nincoming message = ', incoming_message)
    
    if (command=='invalid'):
        sendMessage(getChatID(update), 'Please use a valid command or type /help to know the commands I know. All valid commands start with a slash /. Cheers 🍻')
        return
    
    elif (command == 'q'):
        print(incoming_message)
        answerQuery(incoming_message, update)
    
    elif (command == 'n' or command == 'u' or command == 'd'):
        
        '''
            These are interactive answer-related commands.
            n = next answer (for the same question)
            u = upvote current answer
            d =  downvote current answer
        '''

        if not(getUserID(update) in userid_answers_dict):
            sendMessage(getChatID(update), 'Please use a valid command or type /help to know the commands I know. All valid commands start with a slash /. Cheers 🍻')
        elif command == 'n' and len(userid_answers_dict[getUserID(update)]) > 1:
            userid_answers_dict[getUserID(update)] = giveOneAnswer(getChatID(update), userid_answers_dict[getUserID(update)])
        elif command == 'u' or command == 'd':
            voteOnAnswer(getChatID(update), command, update)
        else:
            sendMessage(getChatID(update), 'No more answers available for this question.')
            
    elif command == 'general':
        sendMessage(getChatID(update), findGeneralAnswer(incoming_message))

    else:
        first_name = update.message.from_user.first_name
        if (update.message):
            first_name = update.message.from_user.first_name
        else:
            first_name = update.edited_message.text.from_user.first_name
        replyToCommand(getChatID(update), command, first_name)

    return


def findGeneralAnswer(ques):

    global small_talk_answers, small_talk_questions, model

    features = model([ques] + small_talk_questions)

    corr = np.inner(features, features)
    
    if max(corr[0][1:]) < 0.4:
        return "I didn't understand that :("

    return random.choice(small_talk_answers[corr[0][1:].argmax()].split('&&'))
    

def findAnswer(features):                  # check database questions for similarity and return suitable answer.
    
    global questions_obj_list, collection_answers

    corr = np.inner(features, features)
    
    answers_obj_list = []

    if max(corr[0][1:]) > 0.4:                      # Similarity threshold
        idx = corr[0][1:].argmax()
        for id in questions_obj_list[idx]['answers']:
            answers_obj_list.append(collection_answers.find_one({'_id': id}))
    
    return answers_obj_list

@tl.job(interval=timedelta(seconds=1800))            # This decorator enables this function to execute every 5 hours
def updateDB():

    '''
        Performs the timely database update
        Updates are fetched from Gsheets API
        This runs on a seperate thread.
    '''

    global questions_obj_list, questions_text_list, model, collection_answers, collection_questions

    gc = gspread.service_account()
    client = MongoClient('localhost', 27017)

    print('\nDatabase Update in Process...\n')
    sht = gc.open_by_url('https://docs.google.com/spreadsheets/d/1Jged2Bis3KymVCfBqdaSUf6mJZm0iJDWai3QxtrkLHQ/edit#gid=0')
    worksht = sht.get_worksheet(0)
    qna_dict = worksht.get_all_records()

    for obj in qna_dict:

        ques = obj['Question']
        ans = obj['Answer']

        features = model([ques] + questions_text_list)
        corr = np.inner(features, features)

        if max(corr[0][1:]) > 0.75:     # new question doesn't need to be inserted
            
            idx = corr[0][1:].argmax()
            ans_id_list = questions_obj_list[idx]['answers']
            ans_obj_list = [collection_answers.find_one({'_id': id}) for id in ans_id_list]
            ans_text_list = [obj['text'] for obj in ans_obj_list]

            for a in ans.split('&&'):
                if not a in ans_text_list and not a.isspace() and a != '':
                    temp_id = collection_answers.insert_one({'text': a, 'upvotes': [], 'downvotes': [], 'score': 0}).inserted_id
                    ans_id_list.append(temp_id)
                    ans_text_list.append(a)

            myquery = { '_id': questions_obj_list[idx]['_id'] }
            newvals = { '$set': { 'answers':  ans_id_list } }
            collection_questions.update_one(myquery, newvals)

        else:                           # new question needs to be inserted
            
            ans_id_list = []
        
            if ans != '' and (not ans.isspace()):
                ans_list_json = [{'text': a, 'upvotes': [], 'downvotes': [], 'score': 0} for a in ans.split('&&')]
                ans_list_id = collection_answers.insert_many(ans_list_json).inserted_ids

            ques_json = {'text': ques, 'answers': ans_id_list}
            collection_questions.insert_one(ques_json)

    
    questions_obj_list = [obj for obj in collection_questions.find()]       # updating global variables
    questions_text_list = [obj['text'] for obj in questions_obj_list]
    
    print('\nDatabase Update Completed Successfully !!\n')


@tl.job(interval=timedelta(seconds=1800))            # This decorator enables this function to execute every 5 hours
def updateCSV():

    '''
        Performs the timely database update
        Updates are fetched from Gsheets API
        This runs on a seperate thread.
    '''

    global small_talk_answers, small_talk_questions

    gc = gspread.service_account()

    print('\nCSV update in progress...\n')
    sht2 = gc.open_by_url('https://docs.google.com/spreadsheets/d/1GW_1AJXxMo50xp92BQWuN26DhTLWWxzUMIN_YMvnojc/edit#gid=0')
    worksht2 = sht2.get_worksheet(0)
    small_dict = worksht2.get_all_records()

    with open('./smalltalk.csv', mode='w', encoding='utf-8-sig', newline='') as smalltalk:
        writer = csv.writer(smalltalk, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Question', 'Answers'])
        for obj in small_dict:

            ques = obj['Question']
            ans = obj['Answers']

            writer.writerow([ques, ans])
    
    small_talk = pd.read_csv('./smalltalk.csv', sep=',')


    small_talk_questions = list(small_talk['Question'])
    small_talk_answers = list(small_talk['Answers'])
    
    print('\nCSV Update Completed Successfully !!\n')


@tl.job(interval=timedelta(seconds=600))            # This decorator enables this function to execute every 30 min
def updateSpreadSheet():

    '''
        Flushes unsered questions to a different spreadsheet.
        This runs on a seperate thread.
    '''

    global unanswered_questions

    gc = gspread.service_account()

    print('\nSpreadSheet update in progress...\n')
    sht = gc.open_by_url('https://docs.google.com/spreadsheets/d/15kexJqY8idu66aJ7kOqClOdYYVfV7x8xDlVp7t6_b8w/edit#gid=0')
    worksht = sht.get_worksheet(0)
    
    for ques in unanswered_questions:
        worksht.append_row([ques])
    
    unanswered_questions = []
    
    print('\nSpreadSheet Update Completed Successfully !!\n')


def signal_handler(sig, frame):
    global tl
    tl.stop()
    sys.exit(0)

small_talk = pd.read_csv('./smalltalk.csv', sep=',')

small_talk_questions = list(small_talk['Question'])
small_talk_answers = list(small_talk['Answers'])

client = MongoClient('localhost', 27017)

print ('\nLoading NLP Model...\n')
print('\nIf you are running it the first time, it might take a few minutes to download the model\n')
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print ('\nNLP model loaded successfully !\n')

db = client['mallya']

collection_questions = db['questions']
collection_answers = db['answers']

questions_obj_list = [obj for obj in collection_questions.find()]
questions_text_list = [obj['text'] for obj in questions_obj_list]

userid_answers_dict = {}                    # this dict stores an list of answers specfic to one user.
unanswered_questions = []                   # this list stores unanswered questions to be flushed into sheets every 30 minutes.

tl.start()

signal.signal(signal.SIGINT, signal_handler)
