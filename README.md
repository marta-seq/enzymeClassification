# enzymeClassification
Recurrent deep neural networks for enzyme functional classification

Enzyme functional annotation has been a challenging problem in Bioinformatics for many years now, with Deep Learning recently appearing as an efficient alternative. Here, the use of recurrent neural networks, trained from sequential data and boosted by the use of attention mechanisms, is analysed. We assess the consequences of the choice of different parameters, as the length of the sequence and type of truncation, often not mentioned in previous studies. We also compare the use of different aminoacid encoding schemes to describe the protein, using one-hot, z-scales and Blosum62 encodings, as well as embedding layers. Lastly, we try to understand what the network is learning and inferring. 


Our results show that for enzyme classification, networks formed with Bidirectional recurrent layers and attention lead to better results. In addition, using simpler encoding schemes (e.g. one-hot) leads to higher performance. Using attention and embedding layers, we demonstrate that the model is capable of learning biological meaningful representations.

If you use this repository for your research please cite the following publication:
ADD
 Recurrent deep neural networks for enzyme functional annotation; Sequeira, AM, Rocha, M. 
 
