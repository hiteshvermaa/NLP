from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('stsb-roberta-large')


file_1 = {"section-1": ["I like Python because I can build AI applications", "There is a garbage on the table"]}
file_2 = {  "section-1": ["There is a dog", "This is the section-1 of file 2"], 
            "section-2": ["I like Python because I can do data analytics", "The table is not clean"]}


for i in range(len(file_1)):
    for j in range(len(file_2)):
        # encode list of sentences to get their embeddings
        embedding1 = model.encode(file_1["section-{}".format(i+1)], convert_to_tensor=True)
        embedding2 = model.encode(file_2["section-{}".format(j+1)], convert_to_tensor=True)
        # compute similarity scores of two embeddings
        cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)

        for k in range(len(file_1["section-{}".format(i+1)])):
            for l in range(len(file_2["section-{}".format(j+1)])):
                # print(cosine_scores[k][l].item())
                if cosine_scores[k][l].item() >= 0.5:
                    print("Sentence 1:", file_1["section-{}".format(i+1)][k])
                    print("Sentence 2:", file_2["section-{}".format(j+1)][l])
                    print("Similarity Score:", cosine_scores[k][l].item())
                    print()

