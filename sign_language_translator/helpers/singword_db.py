import json
from typing import List

class SignWord:
    name: str
    zh_name: str
    phonetic: str
    recognizable: bool
    is_vocabulary: bool
    wordbook_ids: List[int]

    def __init__(
            self,
            class_id: int = -1,
            name: str = None,
            zh_name: str = None,
            phonetic: str = None,
            recognizable: bool = False,
            is_vocabulary: bool = False,
            wordbook_ids: List[int] = []):
        self.class_id = class_id
        self.name = name
        self.zh_name = zh_name
        self.phonetic = phonetic
        self.recognizable = recognizable
        self.is_vocabulary = is_vocabulary
        self.wordbook_ids = wordbook_ids

    @staticmethod
    def to_json(sign_word: 'SignWord'):
        return {
            'class_id': sign_word.class_id,
            'name': sign_word.name,
            'zh_name': sign_word.zh_name,
            'phonetic': sign_word.phonetic,
            'recognizable': sign_word.recognizable,
            'is_vocabulary': sign_word.is_vocabulary,
            'wordbook_ids': sign_word.wordbook_ids
        }

    @staticmethod
    def from_json(json):
        return SignWord(
            -1,
            json['name'],
            json['zh_name'],
            json['phonetic'],
            json['recognizable'],
            json['is_vocabulary'],
            json['wordbook_ids']
        )

# 讀取資料庫
def read_database(file_path: str) -> List[SignWord]:
    with open(file_path, 'r', encoding='utf-8') as file:
        serialized_data  = json.load(file)
        sign_words = []
        for data in serialized_data:
            sign_words.append(SignWord.from_json(data))
        return sign_words

# 寫入資料庫
def write_database(data: SignWord, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('[\n')
        for item in data[:-1]:
            file.write(f'    {json.dumps(SignWord.to_json(item), ensure_ascii=False)},\n')
        file.write(f'    {json.dumps(SignWord.to_json(data[-1]), ensure_ascii=False)}\n')
        file.write(']\n')