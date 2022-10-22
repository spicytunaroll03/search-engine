from tracemalloc import start
from pyparsing import ParseSyntaxException
from ast import Str
from nltk.stem import PorterStemmer
from os import link
import sys
import xml.etree.ElementTree as et
import re
import nltk
import numpy
import file_io
import time
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))
nltk_stemmer = PorterStemmer()


class Indexer:
    def __init__(self):
        self.id_title_dict = {}
        self.title_id_dict = {}
        self.id_pagerank_dict = {}
        self.word_relevance_dict = {}
        self.link_dict = {}  # map of page ids to a page ids

    def index(self, XML_filepath, titles_filepath, docs_filepath, words_filepath):
        # titles_filepath, docs_filepath, words_filepath
        all_pages = et.parse(XML_filepath).getroot()
        word_id_count_dict = {}
        max_word_count_dict = {}
        id_title_link_dict = {}
        start_time = time.time()
        # create dictionary of words to avoid stemming each word

        for page in all_pages:
            # populating the title_dict
            title: str = page.find('title').text.strip()
            page_id: int = int(page.find('id').text)
            self.id_title_dict[page_id] = title
            self.title_id_dict[title] = page_id

            # tokenize/stem text body
            text_body: str = page.find('text').text.strip()
            n_regex = '''\[\[[^\[]+?\]\]|[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+'''
            tokens = re.findall(n_regex, text_body)
            tokens += re.findall(n_regex, title)

            # list of all final words after tokenizing, stemming, and stopping
            page_base_words = []
            # creates an empty set for curr page in the link_dict
            link_set_for_page = set()

            for word in tokens:
                # processing links / properly updating the page_base_words list
                if word[0] == "[" and word[1] == "[":
                    page_base_words = page_base_words + \
                        self.process_link(
                            word, page_id, link_set_for_page, id_title_link_dict, title)

                # processing normal text and updating the page_base_words list
                elif word not in STOP_WORDS:
                    base_word = nltk_stemmer.stem(word)
                    page_base_words.append(base_word)

            # POPULATING THE WORD_ID_COUNT_DICT
            self.make_count_dict(word_id_count_dict, page_base_words, page_id)

            # POPULATING THE MAX_WORD_COUNT_DICT
            most_common_word = max(set(page_base_words),
                                   key=page_base_words.count)
            max_word_count_dict[page_id] = page_base_words.count(
                most_common_word)
        

        # TRY TO POPULATE TF per page as we go instead of looping through entire table again
        self.populate_link_dict(id_title_link_dict)
        # print(self.link_dict)
        self.word_relevance_dict = self.calculate_relevance( \
            word_id_count_dict, max_word_count_dict)

        self.page_rank()
        
        file_io.write_title_file(titles_filepath, self.id_title_dict)
        file_io.write_words_file(words_filepath, self.word_relevance_dict) 
        file_io.write_docs_file(docs_filepath, self.id_pagerank_dict)
        print("runtime: " + str((time.time() - start_time)))
    '''
    Helper method that takes in a link and converts it to a list of words.
    '''

    def process_link(self, link_word: str, curr_page_id: int, link_set: set, id_title_link_dict, curr_title):
        temp_string = ""
        n_regex = '''[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+'''
        word_list = []
        link_title = ""

        link_slice = slice(link_word.find("[") + 2, link_word.find("]"), 1)
        word_slice = slice(link_word.find("|") + 1, link_word.find("]"), 1)
        title_slice = slice(link_word.find("[") + 2, link_word.find("|"), 1)
        
        if "|" in link_word:
            temp_string = link_word[word_slice]  # for parsing the text
            link_title = link_word[title_slice]  # for getting the link title
        else:
            temp_string = link_word[link_slice]  # for parsing the text
            link_title = link_word[link_slice]  # for getting the link title

        # FOR UPDATING THE TEMPORARY LINK_DICT
        # checking if the link is in the corpus 
        # if link_title in self.title_id_dict:
        if curr_title != link_title:
            link_set.add(link_title)  # creating a set of titles
            # updating the values of link_dict
        id_title_link_dict[curr_page_id] = link_set


        # FOR PARSING THE LINKS
        link_tokens = re.findall(n_regex, temp_string)

        for link_word in link_tokens:
            if link_word not in STOP_WORDS:
                base_word = nltk_stemmer.stem(link_word)
                word_list.append(base_word)
        return word_list

    # converts page -> title
    def populate_link_dict(self, temp_link_dict):
        for page_id in temp_link_dict:
            temp_set = set()
            if len(temp_link_dict[page_id]) == 0:
                # add every page id except for the current page_id
                all_ids = set(self.id_title_dict.keys())
                all_ids.remove(page_id)
                self.link_dict[page_id] = all_ids

            for title in temp_link_dict[page_id]:
                if title in self.title_id_dict: 
                    temp_set.add(self.title_id_dict[title])
                    self.link_dict[page_id] = temp_set

    '''
    Method that populates and returns the count dictionary which keeps maps from
    a word to a page id to the occurences of the word'''

    def make_count_dict(self, count_dict, bw_list, page_id):
        # NOW UPDATING THE WORD_ID_COUNT_DICT
        for base_word in bw_list:
            # check if the current word is in the dict
            if base_word not in count_dict:
                # put the word in dict
                count_dict[base_word] = {page_id: 1}
            else:  # when the word is in the dictionary, find the right dictionary and kv pair and update the count
                # if the page_id is in the inner dictionary of the word
                if page_id in count_dict[base_word]:
                    count_dict[base_word][page_id] += 1
                else:
                    count_dict[base_word][page_id] = 1

    '''
    Method that populates and returns the term frequency dictionary by using the
    values from the inputted count_dict and max_dict to calculate and store each
    term frequency value.
    '''

    def populate_tf_dict(self, count_dict, max_dict):
        tf_dict = count_dict  # making a copy of count_dict w/ necessary cij vals
        for word in tf_dict:
            for page_id in tf_dict[word]:
                # storing the current val of the copied dict
                curr_val = tf_dict[word][page_id]
                # replacing the value with the tf value (cij/a)
                tf_dict[word][page_id] = curr_val / max_dict[page_id]
        return tf_dict

    '''
    Method that populates a dictionary 
    '''

    def populate_idf_dict(self, word_count_dict):
        idf_dict = {}
        # calculating the total # of documents
        total_pages = len(self.id_title_dict)
        # populating the dictionary that a term -> # of documents it appears in
        for word in word_count_dict:
            word_doc_count = len(word_count_dict[word])
            idf_dict[word] = numpy.log(total_pages / word_doc_count)
        return idf_dict

    def calculate_relevance(self, word_count_dict, max_dict):
        # POPULATING THE TF_DICT
        tf_dict = self.populate_tf_dict(word_count_dict, max_dict)
        # POPULATING THE IDF_DICT
        idf_dict = self.populate_idf_dict(word_count_dict)
        # MAKING THE RELEVANCE_DICT
        relevance_dict = tf_dict
        for word in relevance_dict:
            for page_id in relevance_dict[word]:
                # getting the tf value for the word
                curr_tf_value = tf_dict[word][page_id]
                # getting the idf value for the word
                idf_value = idf_dict[word]
                relevance_dict[word][page_id] = curr_tf_value * idf_value
        return relevance_dict

    def weight(self):
        weight_dict = {}
        # print(self.link_dict)
        # this is the main case for when the page links to other stuff
        for start_id in self.id_title_dict:
            weight_dict[start_id] = {}             

            for end_id in self.id_title_dict: 
                # if (start_id == end_id):
                #     num_of_links = float(len(self.id_title_dict) - 1)
                #     weight_dict[start_id][end_id] = 0.15/len(self.id_title_dict) + 0.85 * (1/num_of_links)

                if start_id not in self.link_dict: 
                    num_of_links = float(len(self.id_title_dict) - 1)
                    weight_dict[start_id][end_id] = 0.15 / len(self.id_title_dict) + 0.85 * (1/num_of_links)

                if start_id in self.link_dict:
                    links_set = self.link_dict[start_id]
                    num_of_links = len(links_set)

                    if (end_id not in links_set and links_set) or (start_id == end_id):
                        weight_dict[start_id][end_id] = 0.15 / len(self.id_title_dict)
                    else:
                        weight_dict[start_id][end_id] = 0.15 / len(self.id_title_dict) + 0.85 * (1/num_of_links)
            # elif start_id not in self.link_dict:
            # weight_dict[start_id][end_id] = (0.15/len(self.id_title_dict)) + 0.85 * \
            # (1/(num_of_links - 1))
        # add another case for when the page doesn't link to anything.   
        return weight_dict


    def page_rank(self):
        weight_dict = self.weight()
        prev_r_dict = {}
        curr_r_dict = {}
        # initalizing the prev_r_dict and curr_r_dict
        for page in self.id_title_dict:
            prev_r_dict[page] = 0
            curr_r_dict[page] = 1/len(self.id_title_dict)

        while self.euclidean_distance(prev_r_dict, curr_r_dict) > 0.001:
            prev_r_dict = curr_r_dict.copy() 

            for page_id in self.id_title_dict:  # for each page we are updating the rank for per iteration
                curr_r_dict[page_id] = 0 
                # for each page that gives weight to the inputted page (aka each val for a page in the link_dict)
                for page in self.id_title_dict:
                    # print(curr_r_dict[page_id])
                    # print(weight_dict[page][page_id])
                    # print(prev_r_dict[page])
                    curr_r_dict[page_id] = curr_r_dict[page_id] + \
                        weight_dict[page][page_id] * prev_r_dict[page]

        for page in self.id_title_dict:
            self.id_pagerank_dict[page] = curr_r_dict[page] 

    def euclidean_distance(self, prev_r, curr_r) -> float:
        sum = 0.00
        for key in prev_r:
            sum += (curr_r[key] - prev_r[key]) ** 2
        return numpy.sqrt(sum)

# ind = Indexer()
# ind.index("testing_files/PageRankExample3.xml", "titles.txt", "docs.txt", "words.txt")
# print(ind.id_pagerank_dict)

if __name__ == "__main__":
    ind = Indexer()
    ind.index(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
