# Information Retrieval Project

hello everyone! \
As part of our infromation retrievel course, we were asked to write a working search engine for the entire English wikipedia corpus.

### Engine Structure

- search_frontend.py: The flask app for running our engine and getting HTTP requests.
- backend.py: a class in charge of holding each index and dictionary used in the retrival process, uses the similarity_functions.py to calculate query scores.
- similarity_functions.py: A python file with different scoring methods used throughout the engine. 
- Inverted_index_gcp.py: 3 classes that help create, read and iterate the indices.

### Index structures:

- The backend file uses locally stored indexes pointing to posting lists of different building blocks of a wikipedia arcticle (title, text and links). 
- Indices are created using the GCP notebook files and later moved to machine running the search engine. 


### How To Use:

- After creating the different indices and updating the backend class paths, run search_frontend and connect to your local host on port 8080 as described in search_fronted.py.\
Navigate to http://localhost:8080/search?query=enter+query+with+pluses
- Another option is running the engine localy on colab connected to you GCP bucket.\
To do so, upload all the relevent files to your google colab notebook and run it!

This project was made by Ofek Zini and Shahaf Harari