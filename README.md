# Spider-Syn
Towards Robustness of Text-to-SQL Models against Synonym Substitution

Paper published in ACL 2021: [arXiv](https://arxiv.org/abs/2106.01065)

If you only use the dataset, you do not need to install any packages.

## Install & Configure

1. Install pytorch 1.3.1 (or latter) that fits your CUDA version 

2. Install the rest of required packages
    ```
    pip install -r requirements.txt
    ```

3. Run this command to install NLTK punkt.
    ```
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
    ```


## Error in using Bert-Attack
Some of my computers raised a urlopen error `[SSL: CERTIFICATE_VERIFY_FAILED]` when using the Bert-attack model, but some did not. I have not found a unified solution. If you also encounter this problem, you can try the different solutions from the internet.



## Pipeline
We conduct the attack based on the preprocessed Spider dataset where the preprocessed method is under review. We provide the preprocessed dataset in this repository.

There are many pipelines you can generate the synonym substitution examples. We provide two examples here:

1. Generate the synonym substitution examples without the text-to-SQL model. You can generate it using the following commands:

    Glove-based:
    ```
    sh glove_based_synonym_substitution.sh
    ```
    Bert-based:
    ```
    sh bert_based_synonym_substitution.sh
    ```

2. Generate the synonym substitution examples with a text-to-SQL model. We use this method to generate the worst-case. This can be a one or multi round generation process:
    ```
    Generate synonym substitution set1
    -->
    Evaluate the set1
    -->
    Extract the failed attack examples
    -->
    Generate synonym substitution set2 from examples in the previous step
    -->
    Evaluate the set2
    -->
    Extract the failed attack examples
    -->
    ......
    ```
    The method of `Generate synonym substitution set*` is the same as the method of `Generate the synonym substitution examples without the text-to-SQL model`.

    The work of `Evaluate the set*` is done by a text-to-SQL model.

    Our `attack.py` can `extract the failed attack examples` for the [RAT-SQL](https://github.com/Microsoft/rat-sql) model by giving an evaluation result file (`*.eval`) to the `--eval_path` argument. For other text-to-SQL models, you can implement a script to extract.

    The difference between `set1` and `set2` can be the number of substituted words where the `--attack_step` argument in `attack.py` can control this number.

    After one or several rounds, you can combine the generated datasets into one. The `combine_dataset.py` is an example code of combination based on RAT-SQL evaluation results.

When you generate the synonym substitution examples, you can conduct adversarial training.
    




## Cite

If you use the dataset or code in this repo, please cite the following paper:

```
@inproceedings{gan-etal-2021-towards,
    title = "Towards Robustness of Text-to-{SQL} Models against Synonym Substitution",
    author = "Gan, Yujian  and
      Chen, Xinyun  and
      Huang, Qiuping  and
      Purver, Matthew  and
      Woodward, John R.  and
      Xie, Jinxia  and
      Huang, Pengsheng",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.195",
    doi = "10.18653/v1/2021.acl-long.195",
    pages = "2505--2515",
}

```