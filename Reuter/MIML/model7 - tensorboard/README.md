# Wiki-Text-Categorization <br/> <br/>

## Model 7
	* CNN across paragraphs
	* Bag of words model
	* Each word is weighted by it's frequency
	* Paragraph embedding is weighted average of word embedding <br/> <br/>
	
	
## How to run: <br/>
python main.py paragraphLength maxParagraphs filterSizes {= "2-3"} num_filters wordEmbeddingDimension batchSize maxepochs = 400 foldername(where model will be saved) <br/> <br/>

## Evaluation
python Fscore_labelwise.py paragraphLength maxParagraphs filterSizes {= "2-3"} num_filters wordEmbeddingDimension batchSize epoch(epoch no of model which needs to be evaluated) foldername(corresponding to fscore for logs) foldername(where model is saved) <br/> <br/>


## To perform hyperparameter tuning - train the model and evaluation on a grid of parameters <br/>
python gridsearch.py