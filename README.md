# PaDOC-query (PAth with Distance, Origin, and Category constraints)

### Implementation in Python 3.7 for PaDOC. Please check requirements.txt for package dependencies

## Major Codes:

- Query Processing

  - ContractPoINetwork.py: Data structure/class for PoI network construction and various functionalities in network (e.g., kCC indexing create, R-tree build, Dijkstra algrithm)

  - BuildContainer.py: Construct k-Closest Cateogry matrix index from PoI road network, deployed in distributed manner

  - PoIOSMNetwork.py: Build PoI road Network from Road Network (downloaded from OpenStreetMap by default) and PoI database (from TripAdvisor)

  - GreedySearch.py: Implement Origin-first variant (**greedy_process_origin**) and PoI-first variant (**greedy_process_PoI**) algorithms

- Pre-process PoI & Network

  - PoIOSMNetwork.py: Fetch road network from OpenStreetMap and map-matching with PoI dataset
  
  - TripAdvisorCrawler/tripAdvisorCrawler/spiders/tripAdvisorSpider.py: Web HTML crawler for collecting reviews information of attratctions from TripAdvisor (https://www.tripadvisor.com)
  
  - LDALearner.py: Perform Natrual Language Processing, including clean the raw text and train Latent topic based model

## Major Data (Reproducing the experimental results in paper by executing Experiment.py):

  - CONSTANTS.py: List all the constants used in this work such as the number of categories, k in kCC index

  - LDA_Model_6/
    
    -/ldaTrainedModel: Trained learning model (6 categories) used in paper

    -/trainCleanedText.csv: Dataset for training the model which contains 52959 cleaned reviews in U.S. 
  
  - PoI_Network/
  
    - /CSV/: Contain the information of randomly selected PoIs (used in expierments) 
    
    - /Index/: k-CC matrix index constructed in NYC area
    
    - /PKL/: Pickle storage for PoI network
  
  - ExperimentRelated/randomVar.csv: Contains all randomly generated PoI preferences (\theta) used in experiment

#### If you have any question regarding this work, please feel free to reach out to me through xuteng@iastate.edu. Thanks for your interest:)
