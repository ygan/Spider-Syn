import nltk


ALL_JJS={
"most":"most",
"biggest":"big",
"briefest":"brief",
"brightest":"bright",
"broadest":"broad",
"cheapest":"cheap",
"cleverest":"clever",
"closest":"close",
"coldest":"cold",
"dampest":"damp",
"deepest":"deep",
"driest":"dry",
"earliest":"early",
"easiest":"easy",
"extremest":"extreme",
"faintest":"faint",
"fastest":"fast",
"fattest":"fat",
"greatest":"great",
"grimmest":"grim",
"heaviest":"heavy",
"highest":"high",
"hottest":"hot",
"hugest":"huge",
"intensest":"intense",
"largest":"large",
"latest":"late",
"lengthiest":"lengthy",
"lightest":"light",
"littlest":"little",
"longest":"long",
"loudest":"loud",
"lowest":"low",
"mightiest":"mighty",
"newest":"new",
"shallowest":"shallow",
"sharpest":"sharp",
"shortest":"short",
"slightest":"slight",
"slimmest":"slim",
"slowest":"slow",
"smallest":"small",
"smartest":"smart",
"smoothest":"smooth",
"steepest":"steep",
"strongest":"strong",
"sweetest":"sweet",
"thickest":"thick",
"thinnest":"thin",
"tightest":"tight",
"tiniest":"tiny",
"toughest":"tough",
"vaguest":"vague",
"vastest":"vast",
"warmest":"warm",
"weakest":"weak",
"wettest":"wet",
"widest":"wide",
"wisest":"wise",
"worthiest":"worthy",
"tallest":"tall",
"oldest":"old",
"youngest":"young",
}

class MyStemmer():
    def __init__(self):
        self.stemmer = nltk.stem.LancasterStemmer()
    
    def stem(self,w):
        result = w.lower()
        if result == "january":
            return "jan"
        elif result == "february":
            result = "feb"
        elif result == "march":
            return "mar"
        elif result == "april":
            return "apr"
        elif result == "may":
            return "may"
        elif result == "june":
            return "jun"
        elif result == "july":
            return "jul"
        elif result == "august":
            return "aug"
        elif result == "september":
            return "sep"
        elif result == "sept":
            return "sep"
        elif result == "october":
            return "oct"
        elif result == "november":
            return "nov"
        elif result == "december":
            return "dec"
        result = self.stemmer.stem(result)
        if result == "weight":
            result = "weigh"
        if result == "hight":
            result = "high"
        elif result == "won":
            result = "win"
        elif result in ALL_JJS:
            return ALL_JJS[result]
        elif result == "maxim":
            result = "max"
        elif result == "minim":
            result = "min"
        return result




class Tokenizer_Similar_Allennlp():
    def __init__(self, spacy):
        self.spacy = spacy

    def tokenize(self, str_):
        return [tok for tok in self.spacy(str_)]


global_tokenizer = None
global_spacy = None

def get_spacy_tokenizer():
    global global_tokenizer
    global global_spacy

    if global_tokenizer:
        return global_tokenizer

    import spacy
    from spacy.symbols import ORTH, LEMMA
    nlp = spacy.load("en_core_web_sm")
    import re
    from spacy.tokenizer import Tokenizer

    suffixes = nlp.Defaults.suffixes +  (r'((\d{4}((_|-|/){1}\d{2}){2})|((\d{2})(_|-|/)){2}\d{4})(\s\d{2}(:\d{2}){2}){0,1}',) + (r'(\d{1,2}(st|nd|rd|th){0,1}(,|\s)){0,1}((J|j)an(uary){0,1}|(F|f)eb(ruary){0,1}|(M|m)ar(ch){0,1}|(A|a)pr(il){0,1}|(M|m)ay|(J|j)un(e){0,1}|(J|j)ul(y){0,1}|(A|a)ug(ust){0,1}|(S|s)ep(tember){0,1}|(O|o)ct(ober){0,1}|(N|n)ov(ember){0,1}|(D|d)ec(ember){0,1})(\s|,)(\d{1,2}(st|nd|rd|th){0,1}(\s|,){1,3}){0,1}\d{4}',) + ( r'(\d{1,6}(_|-|\+|/)\d{0,6}[A-Za-z]{0,6}\d{0,6}[A-Za-z]{0,6})',)
    suffix_regex = spacy.util.compile_suffix_regex(suffixes)
    nlp.tokenizer.suffix_search = suffix_regex.search
    
    nlp.tokenizer.add_special_case(u'Ph.D', [{ORTH: u'Ph.D', LEMMA: u'ph.d'}])
    nlp.tokenizer.add_special_case(u'id', [{ORTH: u'id', LEMMA: u'id'}])
    nlp.tokenizer.add_special_case(u'Id', [{ORTH: u'Id', LEMMA: u'id'}])
    nlp.tokenizer.add_special_case(u'ID', [{ORTH: u'ID', LEMMA: u'id'}])
    nlp.tokenizer.add_special_case(u'iD', [{ORTH: u'iD', LEMMA: u'id'}])
    nlp.tokenizer.add_special_case(u'statuses', [{ORTH: u'statuses', LEMMA: u'status'}])

    global_tokenizer = Tokenizer_Similar_Allennlp(nlp)
    global_spacy = nlp
    return global_tokenizer