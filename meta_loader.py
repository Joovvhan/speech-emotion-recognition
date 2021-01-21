import textgrid
import csv
from glob import glob
import os
import re

emotions = ['happiness', 'anger', 
            'neutral', 'sadness', 
            'disgust', 'fear', 
            'surprise', 'sleepy']

# emotions = ['anger', 'disgust', 
#             'fear', 'happiness', 
#             'sadness', 'surprise', 
#             'neutral', 'pain', 'sleepy']

def get_savee_meta(file):
    speaker = 'SAVEE_' + file.split('/')[-2]
    pattern_emotion = [('a\d+.wav', 'anger'),
                       ('d\d+.wav', 'disgust'),
                       ('f\d+.wav', 'fear'),
                       ('h\d+.wav', 'happiness'),
                       ('n\d+.wav', 'neutral'),
                       ('sa\d+.wav', 'sadness'),
                       ('su\d+.wav', 'surprise'),
                      ]

    f_name = os.path.basename(file)
    for pattern, emotion in pattern_emotion:
        if re.match(pattern, f_name):
            e = emotion
            break
    return file, speaker, e

def get_tess_meta(file):
    f_name = os.path.basename(file)
    speaker, word, emotion = f_name.rstrip('.wav').split('_') # YAF_youth_disgust.wav
    speaker = f'TESS_{speaker}'
    
    if emotion == 'ps': emotion = 'surprise'
    elif emotion == 'sad': emotion = 'sadness'
    elif emotion == 'angry': emotion = 'anger'
    elif emotion == 'happy': emotion = 'happiness'
    # anger, disgust, fear, happiness, 
    # pleasant surprise, sadness, and neutral
    return file, speaker, emotion

'''
Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
'''

ravdess_code2emo = {
 '01': 'neutral', 
 '02': 'calm', 
 '03': 'happiness', 
 '04': 'sadness', 
 '05': 'anger', 
 '06': 'fear', 
 '07': 'disgust', 
 '08': 'surprise'
}

def get_ravdess_meta(file):
    f_name = os.path.basename(file)
    mod, _, e, intensity, s, r, actor = f_name.rstrip('.wav').split('-') 
    # 03-01-01-01-01-02-03.wav
    speaker = f'RAVDESS_{actor}'
    
    emotion = ravdess_code2emo[e]
    return file, speaker, emotion

crema_d_code2emo = {
    'ANG': 'anger',
    'DIS': 'disgust',
    'FEA': 'fear',
    'HAP': 'happiness',
    'NEU': 'neutral',
    'SAD': 'sadness'
}

def get_crema_d_meta(file):
    f_name = os.path.basename(file)
    # 1001_DFA_ANG_XX.wav
    actor, s, e, intensity = f_name.rstrip('.wav').split('_') 
    
    speaker = f'CREMA-D_{actor}'
    
    emotion = crema_d_code2emo[e]
    return file, speaker, emotion

urdu_code2emo = {
    'A': 'anger',
    'H': 'happiness',
    'N': 'neutral',
    'S': 'sadness'
}

def get_urdu_meta(file):
    f_name = os.path.basename(file)
    # SM7_F4_H076.wav
    actor, _, e = f_name.rstrip('.wav').split('_') 
    
    speaker = f'URDU_{actor}'
    
    for k in urdu_code2emo:
        if k in e:
            emotion = urdu_code2emo[k]
            break
    return file, speaker, emotion

'''
speaker_id(int) - speaker_gender(m or f) - speaker_age(int) - 
spoken_word(int from 0 to 6) - spoken_emotion(int from 0 to 2) - 
record_id(int)
'''

'''
Level 1 is the the standered level, 
it is the way the speaker speaks daily 
where he/she is expressing a neutral emotions,
finally the level 2 emotion, its when the speaker is expressing 
a high level of positive or negative emotions 
(happiness, joy, sadness, anger, etc…
'''

def get_baved_meta(file):
    f_name = os.path.basename(file)
    # 0-m-21-1-2-375.wav
    actor, g, age, s, e, _ = f_name.rstrip('.wav').split('-') 
    
    speaker = f'BAVED_{actor}'

    if e == '0':
        emotion = 'sadness'
    elif e == '1':
        emotion = 'neutral'
    elif e == '2':
        emotion = "unknown"

    return file, speaker, emotion

def get_vivae_meta(file):
    f_name = os.path.basename(file)
    # S01_achievement_low_01.wav
    actor, e, intensity, _ = f_name.rstrip('.wav').split('_') 
    
    speaker = f'VIVAE_{actor}'

    if e == 'achievement':
        emotion = 'sadness'
    elif e == 'pleasure' or e == 'achievement':
        emotion = 'happiness'
    else:
        emotion = e

    return file, speaker, emotion

def get_shemo_meta(file):
    f_name = os.path.basename(file)
    # F03H02.wav
    code = f_name.rstrip('.wav') 
    actor, e, _ = code[0:3],  code[3], code[4:]
    
    speaker = f'ShEMO_{actor}'
    
    if e == 'A':
        emotion = 'anger'
    elif e == 'H':
        emotion = 'happiness'
    elif e == 'N':
        emotion = 'neutral'
    elif e == 'S':
        emotion = 'sadness'
    elif e == 'F':
        emotion = 'fear'
    elif e == 'W':
        emotion = 'surprise'
    
    return file, speaker, emotion

# File naming rule: 
# (Gender)(speaker.ID)_(Emotion)_(Sentence.ID)(session.ID)

def get_jl_corpus_meta(file):
    f_name = os.path.basename(file)
    # female1_apologetic_7b_2.wav
    actor, e, s, _ = f_name.rstrip('.wav').split('_')

    speaker = f'JL_corpus_{actor}'
    
    if e == 'happy': emotion = 'happiness'
    elif e == 'sad': emotion = 'sadness'
    elif e == 'neutral': emotion = 'neutral'
    elif e == 'angry': emotion = 'anger'
    else: emotion = 'unknown'
    
    # happy, sad, excited, neutral, angry
    # encouraging, concerned, assertive, anxious, apologetic
    
    return file, speaker, emotion

cafe_code2emo = {
    'C': 'anger',
    'D': 'disgust',
    'J': 'happiness',
    'N': 'neutral',
    'P': 'fear',
    'S': 'surprise',
    'T': 'sadness',
    
}

def get_cafe_meta(file):
    f_name = os.path.basename(file)
    # 01-D-2-1.wav
    actor, e, intensity, s = f_name.rstrip('.wav').split('-')

    speaker = f'CaFE_{actor}'
    
    '''
    C = Colère		(Anger)
    D = Dégoût		(Disgust)
    J = Joie		(Happiness)
    N = Neutre		(Neutral)
    P = Peur		(Fear)
    S = Surprise	(Surprise)
    T = Tristesse	(Sadness)
    '''
    
    emotion = cafe_code2emo[e]
    
    return file, speaker, emotion

def get_eekk_meta(file):
    tg_file = file.replace('.wav', '.TextGrid')
    tg = textgrid.TextGrid.fromFile(tg_file)
    speaker = f'EEKK'
    
    try:
        e = tg[4].intervals[0].mark
    except IndexError:
        e = tg[3].intervals[0].mark
    
    if e in ['sadness', 'anger', 'neutral', 'joy']:
        if e == 'joy': e = 'happiness'
        emotion = e
    else:
        print(tg_file)
        emotion = None

    return file, speaker, emotion

aesdd_code2emo = {
    'a': 'anger',
    'd': 'disgust',
    'f': 'fear',
    'h': 'happiness',
    's': 'sadness'
}

def get_aesdd_meta(file):
    f_name = os.path.basename(file)
    # a01 (1).wav
    e_s, actor = f_name.rstrip('.wav').split(' ')
    e, s = e_s[0], e_s[1:]
    actor = actor.strip('(').strip(')')
    speaker = f'AESDD_{actor}'
    
    emotion = aesdd_code2emo[e]
    
    return file, speaker, emotion

anad_emotion_dict = dict()

with open('./ANAD/ANAD.csv', 'r') as f:
    reader = csv.reader(f)
    for i, line in enumerate(reader):
        if i == 0:
            continue
        f, emotion = line[0].strip("'"), line[1]
        anad_emotion_dict[f] = emotion

def get_anad_meta(file):
    f_name = os.path.basename(file)
    # a01 (1).wav
    try:
        e = anad_emotion_dict[f_name]
        # Happy,angry, and surprised
        if e == 'happy': emotion = 'happiness'
        elif e == 'angry': emotion = 'anger'
        elif e == 'surprised': emotion = 'surprise'
        else: assert False, f'Invalid Emotion: {e}'
    except KeyError:
        emotion = None
    speaker = 'ANAD'
    
    return file, speaker, emotion

def get_emov_meta(file):
    _, actor, e, f_name = file.split('/')
    # EmoV-DB_sorted/jenie/Angry/anger_393-420_0397.wav
    
    if e == 'Amused': emotion = 'happiness'
    elif e == 'Angry': emotion = 'anger'
    elif e == 'Disgusted': emotion = 'disgust'
    elif e == 'Sleepy': emotion = 'sleepy'
    elif e == 'Neutral': emotion = 'neutral'
    else: assert False, f'Invalid Emotion: {e}'

    speaker = f'EMOV_{actor}'
    
    return file, speaker, emotion
        
def get_acryl_meta(file):
    f_name = os.path.basename(file)
    # acriil_hap_00000002.raw
    _, e, num = f_name.rstrip('.raw').split('_')
    
    if e == 'hap': emotion = 'happiness'
    elif e == 'ang': emotion = 'anger'
    elif e == 'dis': emotion = 'disgust'
    elif e == 'neu': emotion = 'neutral'
    elif e == 'fea': emotion = 'fear'
    elif e == 'sad': emotion = 'sadness'
    elif e == 'sur': emotion = 'surprise'
    else: assert False, f'Invalid Emotion: {e}'

    speaker = f'ACRYL'
    
    return file, speaker, emotion

def get_meta(meta_type):
    
    if meta_type == 'SAVEE':
        savee_files = sorted(glob('SAVEE/AudioData/*/*.wav'))
        meta = list(map(get_savee_meta, savee_files))
    elif meta_type == 'TESS':
        tess_files = sorted(glob('TESS/*.wav'))
        meta = list(map(get_tess_meta, tess_files))
    elif meta_type == 'RAVDESS':
        ravdess_files = sorted(glob('RAVDESS/*/*.wav'))
        meta = list(map(get_ravdess_meta, ravdess_files))
    elif meta_type == 'CREMA-D':
        crema_d_files = sorted(glob('CREMA-D/AudioWAV/*.wav'))
        meta = list(map(get_crema_d_meta, crema_d_files))
    elif meta_type == 'URDU':
        urdu_files = sorted(glob('URDU-Dataset/*/*.wav'))
        meta = list(map(get_urdu_meta, urdu_files))
    elif meta_type == 'BAVED':
        baved_files = sorted(glob('BAVED/*/*.wav'))
        meta = list(map(get_baved_meta, baved_files))
    elif meta_type == 'VIVAE':
        vivae_files = sorted(glob('VIVAE/full_set/*.wav'))
        meta = list(map(get_vivae_meta, vivae_files))
    elif meta_type == 'ShEMO':
        shemo_files = sorted(glob('ShEMO/*/*.wav'))
        meta = list(map(get_shemo_meta, shemo_files))
    elif meta_type == 'JL-corpus':
        jl_corpus_files = sorted(glob('JL corpus/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/*.wav'))
        meta = list(map(get_jl_corpus_meta, jl_corpus_files))
    elif meta_type == 'CaFE':
        cafe_files = sorted(glob('CaFE_48k/*/*/*.wav') + glob('CaFE_48k/*/*.wav'))
        meta = list(map(get_cafe_meta, cafe_files))
    elif meta_type == 'ANAD':
        anad_files = sorted(glob('ANAD/*/*/*.wav'))
        meta = list(filter(lambda x: x[2] != None, map(get_anad_meta, anad_files)))
    elif meta_type == 'EEKK':
        eekk_files = sorted(glob('EEKK/ekorpus/*.wav'))
        meta = list(map(get_eekk_meta, eekk_files))
    elif meta_type == 'AESDD':
        aesdd_files = sorted(glob('Acted Emotional Speech Dynamic Database/*/*.wav'))
        meta = list(map(get_aesdd_meta, aesdd_files))
    elif meta_type == 'EMOV':
        emov_files = sorted(glob('EmoV-DB_sorted/*/*/*.wav'))
        meta = list(map(get_emov_meta, emov_files))
    elif meta_type == 'ACRYL':
        acryl_files = sorted(glob('KoreanEmotionSpeech/*/raw/*.raw'))
        meta = list(map(get_acryl_meta, acryl_files))
    else: assert False, f'Invalid meta type [{meta_type}]'
        
    return meta
    
if __name__ == '__main__':
    
    # https://github.com/SuperKogito/SER-datasets

    # SAVEE
    # English (British)
    # SAVEE FILES 480
    # SAVEE/AudioData/DC/a01.wav
    # 480 British English utterances by 4 males actors.
    # 7 emotions: anger, disgust, fear, happiness, sadness, surprise and neutral.

    print(f'SAVEE FILES {len(savee_files)}')
    print(savee_files[0])
    print()

    # TESS
    # English
    # TESS FILES 2800
    # TESS/OAF_back_angry.wav
    # 2800 recording by 2 actresses.
    # 7 emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral.
    tess_files = sorted(glob('TESS/*.wav'))
    print(f'TESS FILES {len(tess_files)}')
    print(tess_files[0])
    print()

    # RAVDESS
    # English
    # 7356 recordings by 24 actors.
    # Speech file (Audio_Speech_Actors_01-24.zip, 215 MB) contains 1440 files: 
    # 60 trials per actor x 24 actors = 1440.
    # Song file (Audio_Song_Actors_01-24.zip, 198 MB) contains 1012 files: 
    # 44 trials per actor x 23 actors = 1012.
    # RAVDESS FILES 1440
    # RAVDESS/Actor_01/03-01-01-01-01-01-01.wav
    # 7 emotions: calm, happy, sad, angry, fearful, surprise, and disgust
    ravdess_files = sorted(glob('RAVDESS/*/*.wav'))
    print(f'RAVDESS FILES {len(ravdess_files)}')
    print(ravdess_files[0])
    print()

    # CREMA-D
    # English
    # 7442 clip of 12 sentences spoken by 91 actors (48 males and 43 females).
    # CREAMA-D FILES 7442
    # CREMA-D/AudioWAV/1001_DFA_ANG_XX.wav
    # 6 emotions: angry, disgusted, fearful, happy, neutral, and sad
    crema_d_files = sorted(glob('CREMA-D/AudioWAV/*.wav'))
    print(f'CREAMA-D FILES {len(crema_d_files)}')
    print(crema_d_files[0])
    print()

    # URDU
    # Urdu
    # 400 utterances by 38 speakers (27 male and 11 female).
    # URDU FILES 400
    # URDU-Dataset/Angry/SM1_F10_A010.wav
    # 4 emotions: angry, happy, neutral, and sad.
    urdu_files = sorted(glob('URDU-Dataset/*/*.wav'))
    print(f'URDU FILES {len(urdu_files)}')
    print(urdu_files[0])
    print()

    # BAVED
    # Arabic
    # 1935 recording by 61 speakers (45 male and 16 female).
    # BAVED FILES 1935
    # BAVED/0/0-m-21-0-1-105.wav
    baved_files = sorted(glob('BAVED/*/*.wav'))
    print(f'BAVED FILES {len(baved_files)}')
    print(baved_files[0])
    print()

    # VIVAE
    # non-speech, 1085 audio file by ~12 speakers.
    # VIVAE FILES 1085
    # VIVAE/full_set/S01_achievement_low_01.wav
    # non-speech 6 emotions: achievement, anger, fear, pain, pleasure, and surprise 
    # with 3 emotional intensities (low, moderate, strong, peak).
    vivae_files = sorted(glob('VIVAE/full_set/*.wav'))
    print(f'VIVAE FILES {len(vivae_files)}')
    print(vivae_files[0])
    print()

    # ShEMO
    # 3000 semi-natural utterances, equivalent to 3 hours and 25 minutes 
    # of speech data from online radio plays by 87 native-Persian speakers.
    # ShEMO FILES 3000
    # ShEMO/female/F01A01.wav
    # 6 emotions: anger, fear, happiness, sadness, neutral and surprise.
    shemo_files = sorted(glob('ShEMO/*/*.wav'))
    print(f'ShEMO FILES {len(shemo_files)}')
    print(shemo_files[0])
    print()

    # JL corpus
    # 2400 recording of 240 sentences by 4 actors (2 males and 2 females).
    # JL corpus FILES 2400
    # ShEMO/female/F01A01.wav
    # 5 primary emotions: angry, sad, neutral, happy, excited. 
    # 5 secondary emotions: anxious, apologetic, pensive, worried, enthusiastic.
    jl_corpus_files = sorted(glob('JL corpus/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/*.wav'))
    print(f'JL corpus FILES {len(jl_corpus_files)}')
    print(jl_corpus_files[0])
    print()

    # CaFE
    # French (Canadian)
    # 6 different sentences by 12 speakers (6 fmelaes + 6 males).
    # 12 * 6 * (6 * 2 + 1) 
    # CaFE FILES 936
    # CaFE_48k/ColŠre/Faible/01-C-1-1.wav
    # 7 emotions: happy, sad, angry, fearful, surprise, disgust and neutral. 
    # Each emotion is acted in 2 different intensities.
    cafe_files = sorted(glob('CaFE_48k/*/*/*.wav') + glob('CaFE_48k/*/*.wav'))
    print(f'CaFE FILES {len(cafe_files)}')
    print(cafe_files[0])
    print()

    # ANAD
    # Arabic
    # 1384 recording by multiple speakers.
    # ANAD FILES 1420
    # ANAD/1sec_segmented_part1/1sec_segmented_part1/V1_1 (1).wav
    # 3 emotions: angry, happy, surprised.
    anad_files = sorted(glob('ANAD/*/*/*.wav'))
    print(f'ANAD FILES {len(anad_files)}')
    print(anad_files[0])
    print()

    # EEKK
    # Estonian
    # 26 text passage read by 10 speakers.
    # EEKK FILES 1164
    # EEKK/ekorpus/105.wav
    # 4 main emotions: joy, sadness, anger and neutral.
    eekk_files = sorted(glob('EEKK/ekorpus/*.wav'))
    print(f'EEKK FILES {len(eekk_files)}')
    print(eekk_files[0])
    print()

    # AESDD
    # Greek
    # around 500 utterances by a diverse group of actors (over 5 actors) simlating various emotions.
    # AESDD FILES 605
    # Acted Emotional Speech Dynamic Database/anger/a01 (1).wav
    # 5 emotions: anger, disgust, fear, happiness, and sadness.
    aesdd_files = sorted(glob('Acted Emotional Speech Dynamic Database/*/*.wav'))
    print(f'AESDD FILES {len(aesdd_files)}')
    print(aesdd_files[0])
    print()

    # EMOV
    emov_files = sorted(glob('EmoV-DB_sorted/*/*/*.wav'))
    print(f'EMOV FILES {len(emov_files)}')
    print(emov_files[0])
    print()

    # ACRYL
    acryl_files = sorted(glob('KoreanEmotionSpeech/*/raw/*.raw'))
    print(f'ACRYL FILES {len(acryl_files)}')
    print(acryl_files[0])
    print()

    savee_meta = list(map(get_savee_meta, savee_files))
    print('SAVEE Meta')
    print(savee_meta[0])
    
    tess_meta = list(map(get_tess_meta, tess_files))
    print('TESS Meta')
    print(tess_meta[0])
    
    ravdess_meta = list(map(get_ravdess_meta, ravdess_files))
    print('RAVDESS Meta')
    print(ravdess_meta[0])
    
    crema_d_meta = list(map(get_crema_d_meta, crema_d_files))
    print('CREMA-D Meta')
    print(crema_d_meta[0])
    
    urdu_meta = list(map(get_urdu_meta, urdu_files))
    print('URDU Meta')
    print(urdu_meta[0])
    
    baved_meta = list(map(get_baved_meta, baved_files))
    print('BAVED Meta')
    print(baved_meta[0])
    
    vivae_meta = list(map(get_vivae_meta, vivae_files))
    print('VIVAE Meta')
    print(vivae_meta[0])

    shemo_meta = list(map(get_shemo_meta, shemo_files))
    print('ShEMO Meta')
    print(shemo_meta[0])
    
    jl_corpus_meta = list(map(get_jl_corpus_meta, jl_corpus_files))
    print('JL corpus Meta')
    print(jl_corpus_meta[0])
    
    cafe_meta = list(map(get_cafe_meta, cafe_files))
    print('CaFE Meta')
    print(cafe_meta[0])
    
    eekk_meta = list(map(get_eekk_meta, eekk_files))
    print('EEKK Meta')
    print(eekk_meta[0])
    
    aesdd_meta = list(map(get_aesdd_meta, aesdd_files))
    print('AESDD Meta')
    print(aesdd_meta[0])
    
    anad_meta = list(filter(lambda x: x[2] != None, map(get_anad_meta, anad_files)))
    print('ANAD Meta')
    print(anad_meta[0])
    
    emov_meta = list(map(get_emov_meta, emov_files))
    print('EMOV Meta')
    print(emov_meta[0])
    
    acryl_meta = list(map(get_acryl_meta, acryl_files))
    print('ACRYL Meta')
    print(acryl_meta[0])
    