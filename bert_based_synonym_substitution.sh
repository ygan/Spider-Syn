#!/bin/sh
number_of_substitutued_words=1

# Generate preprocessed_dataset/for_attack.json file from preprocessed_dataset/dev.json
python preprocess_for_attack.py --input preprocessed_dataset/dev.json --output preprocessed_dataset/for_attack.json


python attack.py --attack_type "bert" --attack_step $number_of_substitutued_words

if [ $number_of_substitutued_words -eq 1 ]; then
    python combine_dataset.py
else 
    python combine_dataset.py --attack_output  "output_step"$number_of_substitutued_words".json"
fi

# You can find a log.html and output_step1.json in the code root directory
# You can try to midify the value of number_of_substitutued_words 