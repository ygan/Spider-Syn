from utils.utils import MyStemmer


class Schema():
    def __init__(self, _tokenizer, stemmer, table_dict, _concept_word):
        self.column_tokens = [_tokenizer.tokenize(col[1]) for col in table_dict["column_names"]]
        self.table_tokens  = [_tokenizer.tokenize(ta)  for ta in table_dict["table_names"]]
        self.column_tokens_table_idx = [col[0] for col in table_dict["column_names"]]
        
        self.column_tokens_text_str = [col[1] for col in table_dict["column_names"]]
        self.column_tokens_lemma_str = [ " ".join([tok.lemma_ for tok in col]) for col in self.column_tokens ]
        self.column_tokens_lemma_str_tokens = [ col.split(" ") for col in self.column_tokens_lemma_str ]

        self.table_tokens_lemma_str = [ " ".join([tok.lemma_ for tok in col]) for col in self.table_tokens ]
        self.table_word_lemma_set  = set([ tok.lemma_  for col in self.table_tokens for tok in col ])
        self.table_tokens_text_str = table_dict["table_names"]

        self.primaryKey = table_dict["primary_keys"]
        self.foreignKey = list(set([ j for i in table_dict['foreign_keys'] for j in i]))
        self.foreignKeyDict = dict()
        for fk in table_dict['foreign_keys']:
            if fk[0] not in self.foreignKeyDict.keys():
                self.foreignKeyDict[fk[0]] = [fk[1]]
            else:
                self.foreignKeyDict[fk[0]].append(fk[1])
            if fk[1] not in self.foreignKeyDict.keys():
                self.foreignKeyDict[fk[1]] = [fk[0]]
            else:
                self.foreignKeyDict[fk[1]].append(fk[0])

        self.db_id = table_dict["db_id"]
        self.column_types = table_dict['column_types']

        self.table_names_original  = table_dict["table_names_original"]
        self.column_names_original = table_dict["column_names_original"]

        self._concept_word = _concept_word
        self._tokenizer = _tokenizer
        if not stemmer:
            stemmer = MyStemmer()
        self._stemmer = stemmer
        if "same_col_idxs" in table_dict:
            self.same_col_idxs = table_dict["same_col_idxs"]
        else:
            self.same_col_idxs = [[]]*len(table_dict["column_names"])

        
        self.table_col_text  = {-1:set()}
        self.table_col_lemma = {-1:set()}
        self.table_col_nltk  = {-1:set()}
        for i in range(len(table_dict["table_names"])):
            self.table_col_text[i] = set()
            self.table_col_lemma[i] = set()
            self.table_col_nltk[i] = set()
        for col,ocol in zip(self.column_tokens,table_dict["column_names"]):
            for tok in col:
                stem_tmp = stemmer.stem(tok.lower_)
                self.table_col_text[-1].add(tok.lower_)
                self.table_col_lemma[-1].add(tok.lemma_)
                self.table_col_nltk[-1].add(stem_tmp)
                self.table_col_text[ocol[0]].add(tok.lower_)
                self.table_col_lemma[ocol[0]].add(tok.lemma_)
                self.table_col_nltk[ocol[0]].add(stem_tmp)

        self.column_tokens_stem_str = [ " ".join([stemmer.stem(tok.text) for tok in col]) for col in self.column_tokens ]
        for i in range(len(self.column_tokens_stem_str)):
            for j in range(i+1,len(self.column_tokens_stem_str),1):
                if self.column_tokens_stem_str[i] == self.column_tokens_stem_str[j] and self.column_tokens_text_str[i] != self.column_tokens_text_str[j] and self.column_tokens_lemma_str[i] != self.column_tokens_lemma_str[j]:
                    stem_word = self.column_tokens_stem_str[i]
                    for z in range(i,len(self.column_tokens_stem_str),1):
                        if self.column_tokens_stem_str[z] == stem_word:
                            self.column_tokens_stem_str[z] = self.column_tokens_text_str[z]
        

        self.tbl_col_tokens_text_str = {}
        self.tbl_col_tokens_lemma_str = {}
        self.tbl_col_tokens_stem_str = {}
        self.tbl_col_idx_back = {}
        self.tbl_col_tokens_text_str_ori = {}

        self.tbl_col_tokens_text_str[-1] = self.column_tokens_text_str
        self.tbl_col_tokens_lemma_str[-1] = self.column_tokens_lemma_str
        self.tbl_col_tokens_stem_str[-1] = self.column_tokens_stem_str
        self.tbl_col_idx_back[-1] = [i for i in range(len(self.column_tokens_text_str))]
        self.tbl_col_tokens_text_str_ori[-1] = [i[1].lower() for i in self.column_names_original]


        for i in range(len(table_dict["table_names"])):
            self.tbl_col_tokens_text_str[i] = []
            self.tbl_col_tokens_lemma_str[i] = []
            self.tbl_col_tokens_stem_str[i] = []
            self.tbl_col_idx_back[i] = []
            self.tbl_col_tokens_text_str_ori[i] = []

        for tid,text,lemma,stem,cid,cor in zip(self.column_tokens_table_idx, self.column_tokens_text_str, self.column_tokens_lemma_str, self.column_tokens_stem_str, self.tbl_col_idx_back[-1],self.tbl_col_tokens_text_str_ori[-1]):
            if tid >= 0:
                self.tbl_col_tokens_text_str[tid].append(text)
                self.tbl_col_tokens_lemma_str[tid].append(lemma)
                self.tbl_col_tokens_stem_str[tid].append(stem)
                self.tbl_col_idx_back[tid].append(cid)
                self.tbl_col_tokens_text_str_ori[tid].append(cor)