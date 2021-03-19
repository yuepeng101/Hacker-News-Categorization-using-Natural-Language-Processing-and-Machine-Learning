# Hacker-News-Categorization-using-Natural-Language-Processing-and-Machine-Learning

These instructions will get you a copy of the project up and how to run on your local machine for testing purposes.


Here are the list of libraries used by the project:
  1. nltk
  2. matplotlib.pyplot
  3. numpy
  4. pandas
  5. math

So before to run the program, make sure to install them properly.


INSTRUCTIONS OF RUNNING AND TESTS.

Firstly, open the entire file using PyCharm. Then simply run the file 'main.py'.

Then, in the python run command line, it will promot user to input a integer 1 or 2 or 3 as the experiment number. By default(press enter directily) the baseline experiment will be operated. It shows as follows:

	---- Task 1: Extract the data and build the model ----

	1. Dividing dataset.

	Please enter the experiment number (1/2/3), or press enter get baseline experiment: 


If you entered 1,2, or press 'enter', the command line will shows the message as belowe (Note: there might be 1 ~ 3 minutes when step 2 and 3 were processed):


	2. Processing and cleaning dataset ... 

	... Dataset has been cleaned!

	3. Calculating the frequency and the conditional probability.

	4. Building model.

	---- Task 2: Using Naïve Bays Classifier to test and predict dataset ----

	The accuracy is  0.8

	---- Task 3: Experiments with the classifier ----

	======== Program finished! ========


If you entered 3, a graph with two different plot images will show up, and the command line will shows the message as below:
	
	2. Processing and cleaning dataset ... 

	Experiment 3: Infrequent Word Filtering.

	======== Program terminated! ========
	

Finally, when a close message shows up, the output files will be created at the original folder.




