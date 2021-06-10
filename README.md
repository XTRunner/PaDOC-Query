# SPRO-query

Implementation in Python 3.7 for SPRO (Semantically Diverse Path with Range and Origin Constraints). Please check requirements.txt for package dependencies

Major Codes:

-- ParallelContractNetwork.py: Build contraction hierarchies on (POI) road network

-- poiOSMNetwork.py: Build PoI road Network from Road Network (downloaded from OpenStreetMap by default) and POI database (from TripAdvisor)

-- BuildContainer.py: Construct Closest Cateogry Vectors index from POI road network

-- greedySearch.py: Include SPRO-Origin-First and SPRO-POI-First algorithm

-- ldaLearner.py: Perform Natrual Language Processing, including clean the raw text and train Latent topic based model

-- experiment.py: Produce the experimental results in paper, including baselines (random walk with restart and Dijkstra algorithms)

-- tripAdvisorCrawler/tripAdvisorCrawler/spiders/tripAdvisorSpider.py: Web HTML crawler for collecting reviews information of attratctions from TripAdvisor (https://www.tripadvisor.com)

Major Data:

-- ldaModel/ldaTrainedModel: Trained learning model (6 categories) used in paper

-- ldaModel/trainCleanedText.csv: Dataset for training the model

-- ldaModel/NYCleanedText.csv: Cleaned/Stemmed reviews of attractions in New York City

-- poiNetwork: (lat, lng) geolocation of POIs and 576 hotels in New York City, plus POI road network before and after contraction hierarchies

-- experimentRelated: the randomly selected origins and \theta used in paper

If you have any question regarding this work, please feel free to reach out to me through xuteng@iastate.edu. Thanks for your interest:)
