from nltk.stem import WordNetLemmatizer
import torch
import pickle,os,copy
import random,json,argparse
import itertools
from textattack.constraints import Constraint
from textattack.datasets import TextAttackDataset 
from textattack.goal_functions import UntargetedClassification
from textattack.models.wrappers import ModelWrapper
from textattack.transformations import WordSwapEmbedding,WordSwapMaskedLM
from textattack.search_methods import GreedyWordSwapWIR
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod,GreedySearch
from textattack.shared import Attack,utils
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.transformations.word_swap_masked_lm import check_if_subword
from utils.utils import get_spacy_tokenizer,MyStemmer
from utils.Schema import Schema



def get_Schema_Token(table_path):
    tables = json.load(open(table_path,'r'))
    lstem = MyStemmer()
    _tokenizer = get_spacy_tokenizer()
    schemas = dict()
    for table in tables:
        schemas[table["db_id"]] = Schema(_tokenizer,lstem,table,None)
    return schemas

class NamedEntityConstraint(Constraint):
    """ A constraint that ensures `transformed_text` only substitutes named entities from `current_text` with other named entities.
    """
    def __init__(self, path, schemas, compare_against_original):
        super().__init__(compare_against_original)
        _dataset = json.load(open(path,"r"))
        self._dataset = dict()
        self.skip_examples = set()
        for i in range(len(_dataset)):
            if _dataset[i]["or_question"] in self._dataset:
                self._dataset[_dataset[i]["or_question"]] = None
                self.skip_examples.add(_dataset[i]["or_question"])
            else:
                self._dataset[_dataset[i]["or_question"]] = _dataset[i]
                self._dataset[_dataset[i]["or_question"]]["question_tokens"] = get_spacy_tokenizer().tokenize(_dataset[i]["question"])
        self.all_words = pickle.load(open(os.path.join("/home/yj/python/Github/bilayerSQL/data/word/20k.pkl"), 'rb'))
        self.schemas = schemas
        
    def _check_constraint(self, transformed_text, current_text):
        transformed_text = transformed_text.text
        if current_text.text not in self._dataset or not self._dataset[current_text.text]:
            return False
        example = self._dataset[current_text.text]

        if transformed_text == example['or_question']:
            return False

        t_text_toks = transformed_text.split(" ")
        if len(t_text_toks) != len(example['pattern_tok']) or len(t_text_toks) != len(example['question_toks']):
            return False

        different_count = 0
        for ttok,tok,ptok in zip(t_text_toks,example['question_toks'],example['pattern_tok']):
            if ttok != tok and (ptok not in ["ST","STC","SC","COL","TABLE","TABLE-COL","DB"] or tok in ["age","name","names","ids","id"]):
                return False
            if ttok != tok:
                different_count += 1
        if different_count > 3 or different_count == 0:
            return False
            
        t_text_toks = get_spacy_tokenizer().tokenize(transformed_text)
        if len(t_text_toks) != len(example['question_tokens']) or len(t_text_toks) != len(example['pattern_tok']):
            return False
        different_count = 0
        for ttok,tok,ptok in zip(t_text_toks, example['question_toks'], example['question_tokens']):#example['pattern_tok']):
            if ttok.text != tok:
                if ttok.lemma_ not in self.all_words or ttok.lemma_ == tok or not tok.islower() or ttok.lemma_ == ptok.lemma_:
                    return False
                if self.schemas and (ttok.lemma_ in self.schemas[example['db_id']].table_col_lemma[-1] or ttok.lemma_ in self.schemas[example['db_id']].table_col_text[-1] or ttok.lemma_ in self.schemas[example['db_id']].table_word_lemma_set):
                    return False
            if ttok.text != tok and ttok.lemma_ != tok and ttok.text + "s" != tok and ttok.text + "es" != tok and ttok.text + "ed" != tok and ttok.text + "ing" != tok and ttok.text + "eing" != tok and ttok.text != tok + "ing":
                different_count += 1
        if different_count == 0:
            return False
        return True










class SpiderAttackDataset(TextAttackDataset):
    def __init__(
        self,
        path,
        with_db_id = False
    ):
        self._dataset = json.load(open(path,"r"))
        for i in range(len(self._dataset)):
            if with_db_id:
                self._dataset[i] = (self._dataset[i]["or_question"] + " " + self._dataset[i]["db_id"],0)
            else:
                self._dataset[i] = (self._dataset[i]["or_question"],0)
        self.examples = list(self._dataset)




class CustomTensorFlowModelWrapper(ModelWrapper):
    def __init__(self, model, path, attack_step, with_db_id, eval_path=None):
        self.model = model
        self.attack_step = attack_step
        self.transform_times = 0
        _dataset = json.load(open(path,"r"))
        self._dataset = dict()
        self.skip_examples = set()
        for i in range(len(_dataset)):
            key = _dataset[i]["or_question"] + " " + _dataset[i]["db_id"] if with_db_id else _dataset[i]["or_question"]
            if key in self._dataset:
                self._dataset[key] = None
                self.skip_examples.add(key)
            else:
                _dataset[i]["transform"] = True
                self._dataset[key] = _dataset[i]
        self.last_example = None
        if eval_path:
            eval_dataset = json.load(open(eval_path,"r"))
            assert len(eval_dataset['per_item']) == len(_dataset)
            for eval_item,data in zip(eval_dataset['per_item'],_dataset):
                key = data["or_question"] + " " + data["db_id"] if with_db_id else data["or_question"]
                if self._dataset[key]:
                    self._dataset[key]["transform"] =  eval_item['exact']

       
    def __call__(self, text_input_list):
        # once return torch.stack([torch.tensor([1,0],dtype=float)], dim=0) means attack success (model fail to classify)
        # once return torch.stack([torch.tensor([0,1],dtype=float)], dim=0) means attack failed

        if len(text_input_list) == 1 and text_input_list[0] in self._dataset and self._dataset[text_input_list[0]]:
            self.last_example = text_input_list[0]
            self.transform_times = 0
            if self._dataset[self.last_example]["transform"]:
                return torch.stack([torch.tensor([1,0],dtype=float)], dim=0)
            else:
                return torch.stack([torch.tensor([0,1],dtype=float)], dim=0)
        if self.last_example in self.skip_examples: # skip without attack
            return torch.stack([torch.tensor([1,0],dtype=float) for i in range(len(text_input_list))], dim=0)
        
        self.transform_times += 1
        final_preds = []
        for i,text in enumerate(text_input_list):
            if self._dataset[self.last_example]["transform"] and ( self.transform_times >= self.attack_step or self.transform_times >= self._dataset[self.last_example]["transform_max_time"]):
                final_preds.append(torch.tensor([0,1],dtype=float))
            else:
                final_preds.append(torch.tensor([1,0],dtype=float))
        final_preds = torch.stack(final_preds, dim=0)
        return final_preds
    
    def get_grad(self, text_input):
        raise NotImplementedError()

    def generate_example(self, transformed_text, original_text):
        if original_text in self._dataset and self._dataset[original_text]:
            example = copy.deepcopy(self._dataset[original_text])
            example["question"] = transformed_text
            example["question_toks"] = transformed_text.split(" ")
            return example
        return None



class SpiderGoalFunction(UntargetedClassification):

    def get_results(self, attacked_text_list, check_skip=False):
        """For each attacked_text object in attacked_text_list, returns a
        result consisting of whether or not the goal has been achieved, the
        output for display purposes, and a score.

        Additionally returns whether the search is over due to the query
        budget.
        """
        results = []
        if self.query_budget < float("inf"):
            queries_left = self.query_budget - self.num_queries
            attacked_text_list = attacked_text_list[:queries_left]
        random.shuffle(attacked_text_list)

        self.num_queries += len(attacked_text_list)
        model_outputs = self._call_model(attacked_text_list)
        for attacked_text, raw_output in zip(attacked_text_list, model_outputs):
            displayed_output = self._get_displayed_output(raw_output)
            goal_status = self._get_goal_status(
                raw_output, attacked_text, check_skip=check_skip
            )
            goal_function_score = self._get_score(raw_output, attacked_text)
            results.append(
                self._goal_function_result_type()(
                    attacked_text,
                    raw_output,
                    displayed_output,
                    goal_status,
                    goal_function_score,
                    self.num_queries,
                    self.ground_truth_output,
                )
            )
        return results, self.num_queries == self.query_budget












class SpiderWordSwapEmbedding(WordSwapEmbedding):
    def __init__(self, max_candidates=15, embedding_type="paragramcf", **kwargs):
        super().__init__(max_candidates=15, embedding_type="paragramcf", **kwargs)
        self.stemmer = MyStemmer()
        self.real_max_candidates = max_candidates
        self.word_lemma = WordNetLemmatizer()
        self.all_words = pickle.load(open(os.path.join("/home/yj/python/Github/bilayerSQL/data/word/20k.pkl"), 'rb'))


    def _get_transformations(self, current_text, indices_to_modify):
            words = current_text.words
            transformed_texts = []

            for i in indices_to_modify:
                word_to_replace = self.word_lemma.lemmatize(words[i])
                replacement_words = self._get_replacement_words(word_to_replace)
                transformed_texts_idx = []
                for r in replacement_words:
                    if self.stemmer.stem(r) == self.stemmer.stem(word_to_replace):
                        continue
                    if self.word_lemma.lemmatize(r) not in self.all_words:
                        continue
                    transformed_texts_idx.append(current_text.replace_word_at_index(i, r))
                    if len(transformed_texts_idx) >= self.real_max_candidates:
                        break
                transformed_texts.extend(transformed_texts_idx)
                
            random.shuffle(transformed_texts)
            return transformed_texts





def build_embedding_attack(args):
    max_candidates = [0,1,5,15]

    model_wrapper = CustomTensorFlowModelWrapper(None, args.dataset, args.attack_step, args.with_db_id, args.eval_path)
    goal_function = SpiderGoalFunction(model_wrapper)
    transformation = SpiderWordSwapEmbedding(max_candidates=max_candidates[args.attack_step])
    search_method = GreedySearch()

    # Our constraints will be the same as Tutorial 1, plus the named entity constraint
    
    constraints = [
        RepeatModification(),
        StopwordModification(),
        NamedEntityConstraint(args.dataset,get_Schema_Token(args.table_path),True)
        ]

    # Now, let's make the attack using these parameters.
    attack = Attack(goal_function, constraints, transformation, search_method)
    return attack,model_wrapper




class SpiderWordSwapMaskedLM(WordSwapMaskedLM):
    def __init__(self, dataset, method="bae", max_candidates=15, **kwargs):
        super().__init__(method=method, max_candidates=max_candidates, **kwargs)
        self._dataset = dataset
    
    def _encode_text(self, text, original_text = None):
        """Encodes ``text`` using an ``AutoTokenizer``, ``self._lm_tokenizer``.

        Returns a ``dict`` where keys are strings (like 'input_ids') and
        values are ``torch.Tensor``s. Moves tensors to the same device
        as the language model.
        """        
        encoding = self._lm_tokenizer.encode_plus(
            text,
            text_pair = " ".join(self._dataset[original_text]["related_text"]) if original_text in self._dataset else original_text,

            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.to(utils.device) for k, v in encoding.items()}

    def _get_transformations(self, current_text, indices_to_modify):
        # extra_args = {}
        if self.method == "bert-attack":
            current_inputs = self._encode_text(current_text.text,current_text.text)
            with torch.no_grad():
                pred_probs = self._language_model(**current_inputs)[0][0]
            top_probs, top_ids = torch.topk(pred_probs, self.max_candidates)
            id_preds = top_ids.cpu()
            masked_lm_logits = pred_probs.cpu()

        transformed_texts = []

        for i in indices_to_modify:
            word_at_index = current_text.words[i]
            if self.method == "bert-attack":
                replacement_words = self._get_replacement_words(
                    current_text,
                    i,
                    id_preds=id_preds,
                    masked_lm_logits=masked_lm_logits,
                )
            else:
                replacement_words = self._get_replacement_words(current_text, i)
            transformed_texts_idx = []
            for r in replacement_words:
                if r != word_at_index:
                    transformed_texts_idx.append(
                        current_text.replace_word_at_index(i, r)
                    )
            transformed_texts.extend(transformed_texts_idx)

        return transformed_texts

    def _bert_attack_replacement_words(
        self,
        current_text,
        index,
        id_preds,
        masked_lm_logits,
    ):
        """Get replacement words for the word we want to replace using BERT-
        Attack method.

        Args:
            current_text (AttackedText): Text we want to get replacements for.
            index (int): index of word we want to replace
            id_preds (torch.Tensor): N x K tensor of top-K ids for each token-position predicted by the masked language model.
                N is equivalent to `self.max_length`.
            masked_lm_logits (torch.Tensor): N x V tensor of the raw logits outputted by the masked language model.
                N is equivlaent to `self.max_length` and V is dictionary size of masked language model.
        """
        # We need to find which BPE tokens belong to the word we want to replace
        masked_text = current_text.replace_word_at_index(
            index, self._lm_tokenizer.mask_token
        )
        current_inputs = self._encode_text(masked_text.text,current_text.text)
        current_ids = current_inputs["input_ids"].tolist()[0]
        word_tokens = self._lm_tokenizer.encode(
            current_text.words[index], add_special_tokens=False
        )

        try:
            # Need try-except b/c mask-token located past max_length might be truncated by tokenizer
            masked_index = current_ids.index(self._lm_tokenizer.mask_token_id)
        except ValueError:
            return []

        # List of indices of tokens that are part of the target word
        target_ids_pos = list(
            range(masked_index, min(masked_index + len(word_tokens), self.max_length))
        )

        if not len(target_ids_pos):
            return []
        elif len(target_ids_pos) == 1:
            # Word to replace is tokenized as a single word
            top_preds = id_preds[target_ids_pos[0]].tolist()
            replacement_words = []
            for id in top_preds:
                token = self._lm_tokenizer.convert_ids_to_tokens(id)
                if utils.is_one_word(token) and not check_if_subword(token):
                    replacement_words.append(token)
            return replacement_words
        else:
            # Word to replace is tokenized as multiple sub-words
            top_preds = [id_preds[i] for i in target_ids_pos]
            if len(top_preds) > 2:
                return []
            products = itertools.product(*top_preds)
            combination_results = []
            # Original BERT-Attack implement uses cross-entropy loss
            cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
            target_ids_pos_tensor = torch.tensor(target_ids_pos)
            word_tensor = torch.zeros(len(target_ids_pos), dtype=torch.long)
            for bpe_tokens in products:
                for i in range(len(bpe_tokens)):
                    word_tensor[i] = bpe_tokens[i]

                logits = torch.index_select(masked_lm_logits, 0, target_ids_pos_tensor)
                loss = cross_entropy_loss(logits, word_tensor)
                perplexity = torch.exp(torch.mean(loss, dim=0)).item()
                word = "".join(
                    self._lm_tokenizer.convert_ids_to_tokens(word_tensor)
                ).replace("##", "")
                if utils.is_one_word(word):
                    combination_results.append((word, perplexity))
            # Sort to get top-K results
            sorted(combination_results, key=lambda x: x[1])
            top_replacements = [
                x[0] for x in combination_results[: self.max_candidates]
            ]
            return top_replacements




class SpiderGreedyWordSwapWIR(GreedyWordSwapWIR):
    def __init__(self, dataset, schemas, attack_step, wir_method="unk"):
        self.wir_method = wir_method
        self._dataset = dataset
        self.attack_step = attack_step
        self.schemas = schemas

    def combine_result(self, results_list, initial_result):
        words = initial_result.attacked_text.words
        new_words = []
        for i,w in enumerate(words):
            ww = w
            for r in results_list:
                if r and r.attacked_text._words[i] != w:
                    ww = r.attacked_text._words[i]
                    results_list[0].attacked_text = results_list[0].attacked_text.replace_word_at_index(i, ww)
                    break
        return results_list[0]

    def _perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text

        # Sort words by order of importance
        # index_order, search_over = self._get_index_order(attacked_text)
        # random.shuffle(index_order)
        if initial_result.attacked_text.text in self._dataset and self._dataset[initial_result.attacked_text.text]:
            index_order = [i for i in range(len(initial_result.attacked_text.words))]
            random.shuffle(index_order)
            available_words = self._dataset[initial_result.attacked_text.text]['transform_text_token']
            attack_step = self.attack_step if self.attack_step < self._dataset[initial_result.attacked_text.text]['transform_max_time'] else self._dataset[initial_result.attacked_text.text]['transform_max_time']
            search_over = False
        else:
            search_over = True

        i = 0
        cur_result = initial_result
        results = None
        results_list = [None] * attack_step
        for j in range(attack_step):
            while i < len(index_order) and not search_over:
                if initial_result.attacked_text.words[index_order[i]] not in available_words:
                    i += 1
                    continue
                transformed_text_candidates = self.get_transformations(
                    # cur_result.attacked_text,
                    initial_result.attacked_text,
                    original_text=initial_result.attacked_text,
                    indices_to_modify=[index_order[i]],
                )
                i += 1
                if len(transformed_text_candidates) == 0:
                    continue
                results, search_over = self.get_goal_results(transformed_text_candidates)
                results = sorted(results, key=lambda x: -x.score)
                # Skip swaps which don't improve the score
                if results[0].score > cur_result.score or results[0].score == 1:
                    cur_result = results[0]
                else:
                    continue
                # If we succeeded, return the index with best similarity.
                if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    best_result = cur_result
                    # @TODO: Use vectorwise operations
                    max_similarity = -float("inf")
                    for result in results:
                        if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                            break
                        candidate = result.attacked_text
                        try:
                            similarity_score = candidate.attack_attrs["similarity_score"]
                        except KeyError:
                            # If the attack was run without any similarity metrics,
                            # candidates won't have a similarity score. In this
                            # case, break and return the candidate that changed
                            # the original score the most.
                            break
                        if similarity_score > max_similarity:
                            max_similarity = similarity_score
                            best_result = result
                    # return best_result
                    results_list[j] = best_result
                    break
        if results_list and results_list[0]:
            if attack_step == 1:
                return results_list[0]
            else:
                return self.combine_result(results_list,initial_result)
        else:
            return cur_result


def build_bert_attack(args):
    from textattack.constraints.overlap import MaxWordsPerturbed
    from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder

    schemas = get_Schema_Token(args.table_path)
    
    model_wrapper = CustomTensorFlowModelWrapper(None, args.dataset, 1, args.with_db_id, args.eval_path)
    goal_function = SpiderGoalFunction(model_wrapper)
    search_method = SpiderGreedyWordSwapWIR(model_wrapper._dataset, schemas, args.attack_step, wir_method="unk")

    constraints = [
        RepeatModification(),
        StopwordModification(),
        NamedEntityConstraint(args.dataset,schemas,True),
        MaxWordsPerturbed(max_percent=0.4),
        UniversalSentenceEncoder(
            threshold=0.2,
            metric="cosine",
            compare_against_original=True,
            window_size=None,
        )
        ]

    transformation = SpiderWordSwapMaskedLM(model_wrapper._dataset, method="bert-attack", max_candidates=48)

    # Now, let's make the attack using these parameters.
    attack = Attack(goal_function, constraints, transformation, search_method)
    return attack,model_wrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',   default="preprocessed_dataset/for_attack.json")
    parser.add_argument('--table_path', default="preprocessed_dataset/tables.json")
    parser.add_argument('--eval_path', default=None, help="RATSQL eval result file")
    parser.add_argument('--output',    default='output', help="Output path is (here + '_step' + '--attack_step' + '.json')")
    parser.add_argument('--output_only_transformation',  action='store_true', default=True, help="Whether remove the unchange examples")
    parser.add_argument('--attack_step', default=1, type=int, help="The number of words be substituted")
    parser.add_argument('--with_db_id',  action='store_true', default=False)
    parser.add_argument('--attack_type', default='bert', help="bert or glove")
    
    args = parser.parse_args()
    if args.attack_step <= 0:
        args.attack_step = 1
    elif args.attack_step > 3:
        args.attack_step = 3
    args.output = args.output + "_step" + str(args.attack_step) + ".json"
    
    dataset = SpiderAttackDataset(args.dataset, args.with_db_id)

    if args.attack_type == "bert":
        attack,model_wrapper = build_bert_attack(args)
    else:
        attack,model_wrapper = build_embedding_attack(args)

    from textattack.loggers import CSVLogger # tracks a dataframe for us.
    from textattack.attack_results import SuccessfulAttackResult

    results_iterable = attack.attack_dataset(dataset)
    logger = CSVLogger(color_method='html')
    new_dataset = []

    num_successes = 0
    for result in results_iterable:
        transformation_sucess=False
        if isinstance(result, SuccessfulAttackResult):
            #result.original_result.score is the score in error predition. The lower the prediction is better.
            example = model_wrapper.generate_example(result.perturbed_result.attacked_text.text, result.original_result.attacked_text.text)
            if example:
                new_dataset.append(example)
            logger.log_attack_result(result)
            num_successes += 1
            transformation_sucess=True
        if not args.output_only_transformation and (not transformation_sucess or result.original_result.score >= 0.5): # means this example can not pass test
            example = model_wrapper.generate_example(result.original_result.attacked_text.text, result.original_result.attacked_text.text)
            if example:
                new_dataset.append(example)
            
    from IPython.core.display import display, HTML
    f = open('log.html', 'w')
    f.write(HTML(logger.df[['original_text', 'perturbed_text']].to_html(escape=False))._repr_html_())
    f.close()

    json.dump(new_dataset,open(args.output,'w'), indent=2)
    print("attack dataset length:{}".format(len(new_dataset)))