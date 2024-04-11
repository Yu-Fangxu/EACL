
import json
import logging

import pickle
import random

import pandas as pd
import torch
import vocab
from tqdm import tqdm

def pad_to_len(list_data, max_len, pad_value):
    list_data = list_data[-max_len:]
    len_to_pad = max_len - len(list_data)
    pads = [pad_value] * len_to_pad
    list_data.extend(pads)
    return list_data

def get_emorynlp_vocabs(file_paths):
    emotion_vocab = vocab.Vocab()
    emotion_vocab.word2index('neutral', train=True)
    for file_path in file_paths:
        data = json.load(open(file_path, 'r'))
        for episode in tqdm(data['episodes'],
                        desc='processing file {}'.format(file_path)):
            for scene in episode['scenes']:
                for utterance in scene['utterances']:
                    emotion = utterance['emotion'].lower()
                    emotion_vocab.word2index(emotion, train=True)
    torch.save(emotion_vocab.to_dict(), './data/EmoryNLP/label_vocab.pkl')
    logging.info('total {} emotions'.format(len(emotion_vocab)))

def get_meld_vocabs(file_paths):
    emotion_vocab = vocab.Vocab()
    emotion_vocab.word2index('neutral', train=True)
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        for row in tqdm(data.iterrows(),
                        desc='get vocab from {}'.format(file_path)):
            meta = row[1]
            emotion = meta['Emotion'].lower()
            emotion_vocab.word2index(emotion, train=True)
    torch.save(emotion_vocab.to_dict(), "./erc/data/MELD/label_vocab.pkl")
    logging.info('total {} emotions'.format(len(emotion_vocab)))


def build_dataset(dialogues, train=False):
    ret_utterances = []
    ret_labels = []
    for dialogue in dialogues:
        utterance_ids = []
        query = 'For utterance:'
        query_ids = tokenizer(query)['input_ids'][1:-1]
        for idx, turn_data in enumerate(dialogue):
            text_with_speaker = turn_data['speaker'] + ':' + turn_data['text']
            token_ids = tokenizer(text_with_speaker)['input_ids'][1:]
            utterance_ids.append(token_ids)
            if turn_data['label'] < 0:
                continue
            full_context = [CONFIG['CLS']]
            lidx = 0
            for lidx in range(idx):
                total_len = sum([len(item) for item in utterance_ids[lidx:]]) + 8
                if total_len + len(utterance_ids[idx]) <= CONFIG['max_len']:
                    break
            lidx = max(lidx, idx-8)
            for item in utterance_ids[lidx:]:
                full_context.extend(item)

            query_idx = idx
            prompt = dialogue[query_idx]['speaker'] + ' feels <mask>'
            full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:]
            input_ids = full_context + full_query
            input_ids = pad_to_len(input_ids, CONFIG['max_len'], CONFIG['pad_value'])
            ret_utterances.append(input_ids)
            ret_labels.append(dialogue[query_idx]['label'])

            if train and idx > 3 and torch.rand(1).item() < 0.2:
                query_idx = random.randint(lidx, idx-1)
                if dialogue[query_idx]['label'] < 0:
                    continue
                prompt = dialogue[query_idx]['speaker'] + ' feels <mask>'
                full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:]
                input_ids = full_context + full_query
                input_ids = pad_to_len(input_ids, CONFIG['max_len'], CONFIG['pad_value'])
                ret_utterances.append(input_ids)
                ret_labels.append(dialogue[query_idx]['label'])
            
    dataset = TensorDataset(
        torch.LongTensor(ret_utterances),
        torch.LongTensor(ret_labels)
    )
    return dataset

def get_iemocap_vocabs(file_paths):
    emotion_vocab = vocab.Vocab()
    emotion_vocab.word2index('neu', train=True)
    for file_path in file_paths:
        data = json.load(open(file_path, 'r'))
        for dialog in tqdm(data,
                desc='get vocab from {}'.format(file_path)):
            for utterance in dialog:
                emotion = utterance.get('label')
                if emotion is not None:
                    emotion_vocab.word2index(emotion, train=True)
    
    torch.save(emotion_vocab.to_dict(), './data/IEMOCAP/label_vocab.pkl')
    logging.info('total {} emotions'.format(len(emotion_vocab)))

def load_emorynlp_turn(file_path):
    with open('./data/EmoryNLP/label_vocab.pkl', 'rb') as f:
        emotion_vocab = pickle.load(f)
    data = json.load(open(file_path, 'r'))
    dialogues = []
    speaker_vocab = vocab.Vocab()
    for episode in tqdm(data['episodes'],
                    desc='processing file {}'.format(file_path)):
        for scene in episode['scenes']:
            dialogue = []
            for utterance in scene['utterances']:
                text = utterance['transcript']
                speaker = utterance['speakers'][0]
                speaker = speaker.split(' ')[0]
                emotion = utterance['emotion'].lower()
                emotion_idx = emotion_vocab[emotion]
                turn_data = {}
                turn_data['speaker'] = speaker
                speaker_vocab.word2index(speaker, train=True)
                turn_data['text'] = text
                turn_data['label'] = emotion_idx
                turn_data['emotion'] = emotion
                dialogue.append(turn_data)
            dialogues.append(dialogue)
    return dialogues


def load_meld_turn(file_path):
    with open('./data/MELD/label_vocab.pkl', 'rb') as f:
        emotion_vocab = pickle.load(f)
    data = pd.read_csv(file_path)
    pre_dial_id = -1
    dialogues = []
    dialogue = []
    speaker_vocab = vocab.Vocab()
    for row in tqdm(data.iterrows(),
                    desc='processing file {}'.format(file_path)):
        meta = row[1]
        text = meta['Utterance'].replace('’', '\'').replace("\"", '')
        speaker = meta['Speaker']
        emotion = meta['Emotion'].lower()
        emotion_idx = emotion_vocab[emotion]# emotion_vocab.word2index(emotion)
        turn_data = {}
        turn_data['speaker'] = speaker
        speaker_vocab.word2index(speaker, train=True)
        turn_data['text'] = text
        turn_data['label'] = emotion_idx

        dialogue_id = meta['Dialogue_ID']
        if pre_dial_id == -1:
            pre_dial_id = dialogue_id
        if dialogue_id != pre_dial_id:
            dialogues.append(dialogue)
            dialogue = []
        pre_dial_id = dialogue_id
        dialogue.append(turn_data)
    dialogues.append(dialogue)

    return dialogues

def load_iemocap_turn(file_path):
    with open('./data/IEMOCAP/label_vocab.pkl', 'rb') as f:
        emotion_vocab = pickle.load(f)
    data = json.load(open(file_path, 'r'))
    speaker_pools = json.load(open('./data/IEMOCAP/name_pool', 'r'))
    dialogues = []
    count = 0
    for dialog in tqdm(data,
            desc='processing file {}'.format(file_path)):
        dialogue = []
        t_vocab = vocab.Vocab()
        speaker_vocab = vocab.Vocab()
        for utterance in dialog:
            speaker = utterance.get('speaker').upper()
            text = utterance.get('text').replace('[LAUGHTER]', '')
            emotion = utterance.get('label')
            speaker = speaker_pools[t_vocab.word2index(speaker, train=True)]
            speaker_vocab.word2index(speaker, train=True)
            turn_data = {}
            turn_data['speaker'] = speaker
            turn_data['text'] = text
            turn_data['emotion'] = emotion
            if emotion is not None:
                emotion_idx = emotion_vocab[emotion]
                count += 1
            else:
                emotion_idx = -1
            turn_data['label'] = emotion_idx
            
            dialogue.append(turn_data)
        dialogues.append(dialogue)
    print(count)
    return dialogues

def load_dailydialog_turn(file_path):
    with open('./data/DailyDialog/label_vocab.pkl', 'rb') as f:
        emotion_vocab = pickle.load(f)
    f = open(file_path, 'r')
    data = f.readlines()
    f.close()
    dialogues = []
    dialogue = []
    speaker_vocab = vocab.Vocab()
    for utterance in tqdm(data,
                    desc='processing file {}'.format(file_path)):
        if utterance == '\n':
            dialogues.append(dialogue)
            dialogue = []
            continue
        speaker = utterance.strip().split('\t')[0]
        text = ' '.join(utterance.strip().split('\t')[1:-1])
        emotion = utterance.strip().split('\t')[-1]
        emotion_idx = emotion_vocab[emotion]
        turn_data = {}
        turn_data['speaker'] = speaker
        speaker_vocab.word2index(speaker, train=True)
        turn_data['text'] = text
        turn_data['label'] = emotion_idx
        turn_data['emotion'] = emotion
        dialogue.append(turn_data)
    return dialogues