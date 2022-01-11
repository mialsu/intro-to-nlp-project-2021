# intro-to-nlp-project-2021

This project was course assignment of Univerity of Turku course Introduction to Natural Language Processing. The aim was to classify short Finnish sentences by whether they expressed anger or joy. The dataset was labelled by course participants during the course.

After processing data in dataprocessor.py a simple SVM-classifier was trained but results weren't encouraging at all. With that in mind a neural network classifier was trained and after some parameter tuning and plotting results validation loss of about 0.29 and validation accuracy of 91.8% were achieved respectively.

The usage of simple bag-of-words -method could be replaced with Tfidf-vectorization and through that higher accuracy could be reached. Also it would be possible to alter the neural network structure and possibly get performance increases. The dataset had it's whole vocabulary in the training phase and no stopwords or other words were dropped, so that could also be an upgrade.

All in all the accuracy of 91.8% is acceptable and further development is still possible.

## Usage
Run dataprocessor.py first to create dataset for model training part.
