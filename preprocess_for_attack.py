import json,argparse


def refresh_sql(sql,new_set,total,i):
    tmp_set = set(total[i]["transform_text_token"])
    if (not new_set or new_set == old_set) and tmp_set.intersection(old_set):
        sql["related_text"].append(total[i]["question"])
        new_set = tmp_set
    elif new_set and tmp_set.intersection(old_set) and tmp_set.intersection(old_set) != new_set.intersection(old_set):
        sql["related_text"].append(total[i]["question"])
    return new_set,sql

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   default="preprocessed_dataset/dev.json")
    parser.add_argument('--output', default="preprocessed_dataset/for_attack.json")
    args = parser.parse_args()

    total = json.load(open(args.input,"r"))
    count = 0
    for j,sql in  enumerate(total):
        pre_tok = None
        count = 0
        change_list = []
        for (i,tok),ptok in zip(enumerate(sql['question_toks']),sql['pattern_tok']):
            if pre_tok != ptok and ptok in ["ST","STC","SC","COL","TABLE","TABLE-COL","DB"] and tok not in ["age","ages","year","years","name","names","ids","id"] and tok.islower():
                count += 1
                change_list.append(tok)
            pre_tok = ptok
        sql["transform_max_time"] = count
        sql["transform_text_token"] = change_list
        sql["or_question"] = sql["question"]


    for j,sql in  enumerate(total):
        new_set = set()
        old_set = set(sql["transform_text_token"])
        sql["related_text"] = []
        
        for i in range(1,25):
            if j+i < len(total) and total[j+i]["db_id"] == sql["db_id"]:
                new_set,sql = refresh_sql(sql,new_set,total,j+i)
            if len(sql["related_text"]) == 2:
                break
            if j-i >= 0 and total[j-i]["db_id"] == sql["db_id"]:
                new_set,sql = refresh_sql(sql,new_set,total,j-i)
            if len(sql["related_text"]) == 2:
                break

    json.dump(total,open(args.output,"w"),indent=2)