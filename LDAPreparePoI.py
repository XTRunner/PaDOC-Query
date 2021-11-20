from gensim import models
from gensim.test.utils import datapath

from LDALearner import places_reviews

import os, csv
import CONSTANTS


def main():
    folder_name = "LDA_Model_" + str(CONSTANTS.CATEGORY_NUM)

    if os.path.exists(folder_name + "/NYCleanedText.csv"):
        print("Start Loading Cleaned Text...")

        with open(folder_name + "/NYCleanedText.csv", 'r') as rhandle:
            rfile = csv.reader(rhandle, delimiter='|')

            reviews = {}

            for each_row in rfile:
                reviews[each_row[0]] = reviews.get(each_row[0], []) + [each_row[1]]
    else:
        reviews = places_reviews("TripAdvisorCrawler/attractionNYTripAdvisor.csv")

        with open(folder_name + "/NYCleanedText.csv", 'a', newline='') as whandle:
            spamwriter = csv.writer(whandle, delimiter='|')

            for k, v in reviews.items():
                for each_review in v:
                    spamwriter.writerow([k, each_review])

    print("----------------------------------------")

    if not os.path.exists(folder_name + "/NYDivVector.csv"):
        print("Start Generating Div Vector...")

        counter = [0] * CONSTANTS.CATEGORY_NUM

        model_file = datapath(os.getcwd() + "\\" + folder_name + "\\ldaTrainedModel")
        trained_model = models.LdaModel.load(model_file)
        dict_word = trained_model.id2word

        with open(folder_name + "/NYDivVector.csv", 'a', newline='') as whandle:
            spamwriter = csv.writer(whandle)

            for k, v in reviews.items():
                corpus = dict_word.doc2bow((" ".join(v)).split())
                lda_score = [x[1] for x in trained_model[corpus]]

                category_idx = lda_score.index(max(lda_score))
                counter[category_idx] += 1

                spamwriter.writerow([k] + lda_score)

        print("Num of PoIs in each category: ", counter)


if __name__ == "__main__":
    main()