import json,argparse,os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_attack_step',default=1, type=int)
    parser.add_argument('--original_file',  default="preprocessed_dataset/for_attack.json", type=str)
    parser.add_argument('--attack_output',  default="output", type=str)
    parser.add_argument('--eval_base_path', default=None, type=str)
    parser.add_argument('--dataset_output', default="output_combine.json", type=str)
    args = parser.parse_args()

    assert args.max_attack_step >= 1

    question_set = set()
    total = []
    for i in range(args.max_attack_step,-1,-1):
        if i == args.max_attack_step:
            if args.attack_output.endswith(".json"):
                total = json.load(open(args.attack_output,"r"))
            else:
                total = json.load(open(args.attack_output + "_step" + str(i) + ".json" ,"r"))
            for item in total:
                question_set.add(item["or_question"])
            continue
        elif i == 0:
            data_tmp = json.load(open(args.original_file,"r"))
            if args.eval_base_path:
                eval_dataset = json.load(open(args.eval_base_path,"r"))
                assert len(eval_dataset['per_item']) == len(data_tmp)
                for eval_item,data in zip(eval_dataset['per_item'],data_tmp):
                    if not eval_item['exact'] or data["or_question"] not in question_set:
                        total.append(data)
            else:
                for data in data_tmp:
                    if data["or_question"] not in question_set: 
                        total.append(data)
        else:
            data_tmp = json.load(open(args.attack_output + "_step" + str(i) + ".json" ,"r"))
            eval_dataset = json.load(open(args.eval_base_path+ "_step" + str(i),"r"))
            assert len(eval_dataset['per_item']) == len(data_tmp)
            for eval_item,data in zip(eval_dataset['per_item'],data_tmp):
                if not eval_item['exact']:
                    total.append(data)
    
    print(len(total))
    json.dump(total,open(args.dataset_output,'w'), indent=2)