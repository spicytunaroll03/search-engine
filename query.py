from email.mime import base
from pickle import STOP
import sys
import time
from regex import P
import file_io
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))
nltk_stemmer = PorterStemmer()

class Query:
    def __init__(self, titles_path, docs_path, words_path): 
        try: 
            # initializing title file 
            self.id_to_title = {}
            file_io.read_title_file(titles_path, self.id_to_title)
            # initializing words file 
            self.id_page_rank_dict = {}
            file_io.read_docs_file(docs_path, self.id_page_rank_dict) 
            # initializing docs file 
            self.word_relevance_dict = {}
            file_io.read_words_file(words_path, self.word_relevance_dict)
        except:
            raise FileNotFoundError


    def query(self):
        total_sum_dict = {}
        # self.word_relevance_dict[user_input[0]]

        if (len(cmd_input) == 5 and (cmd_input[1] != "--pagerank")) or (len(cmd_input) != 4 and (cmd_input[1] != "--pagerank")):
            raise Exception("Incorrect arguments")

        base_text = user_input.strip()
        n_regex = '''\[\[[^\[]+?\]\]|[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+'''
        base_input_words = re.findall(n_regex, base_text)
        
        tokens = []
        for word in base_input_words:
            if word not in STOP_WORDS:
                tokens.append(nltk_stemmer.stem(word))

        total_sum_dict = {}
        for word in tokens: 
            if word in self.word_relevance_dict:
                for page in self.word_relevance_dict[word]:
                    if cmd_input[1] == "--pagerank ":
                        if page in total_sum_dict:
                            total_sum_dict[page] += self.word_relevance_dict[word][page] * self.id_page_rank_dict[page]
                        else: 
                            total_sum_dict[page] = self.word_relevance_dict[word][page] * self.id_page_rank_dict[page]
                    else:
                        if page in total_sum_dict: 
                            total_sum_dict[page] += self.word_relevance_dict[word][page] 
                        else:
                            total_sum_dict[page] = self.word_relevance_dict[word][page]
        

        kv_pair_list = list(total_sum_dict.items())
        kv_pair_list.sort(key=lambda i:i[1],reverse=True)
        pair_list = kv_pair_list[:10]

        top_ten_pages = [] 

        for kv_pair in pair_list:
            title = self.id_to_title[kv_pair[0]]
            top_ten_pages.append(title)

        return top_ten_pages
        

if __name__ == "__main__":
    while (True):
        cmd_input = sys.argv 
        user_input = input("search> ")
        if user_input == ":quit":
            break
        else:
            if "--pagerank" in cmd_input:
                q = Query(cmd_input[2], cmd_input[3], cmd_input[4])
            else:
                q = Query(cmd_input[1], cmd_input[2], cmd_input[3])

            q.query()    
            top_ten_list = q.query()
            #run through list that query returns list
            title_number = 1
            for title in top_ten_list:
                print(str(title_number) + ". " + title + "\n")
                title_number += 1 