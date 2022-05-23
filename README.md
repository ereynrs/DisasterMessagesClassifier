# Disaster Response Pipeline Project

## Project Summary
The _Disaster Response_ web app enables an emergency worker to classify text messages into the following non exclusive categories - i.e.: a single message can be classified into more than one category - :
1. Related
2. Request
3. Offer
4. Aid Related
5. Medical Help
6. Medical Products
7. Search And Rescue
8. Security
9. Military
10. Child Alone
11. Water
12. Food
13. Shelter
14. Clothing
15. Money 
16. Missing People
17. Refugees
18. Death
19. Other Aid
20. Infrastructure Related
21. Transport
22. Buildings
23. Electricity
24. Tools
25. Hospitals
26. Shops
27. Aid Centers
28. Other Infrastructure
29. Weather Related
30. Floods
31. Storm
32. Fire
33. Earthquake
34. Cold
35. Other Weather
36. Direct Report

The categorization is based on a machine learning model trained with a dataset of real messages that were sent during disaster events. The web app depicts the contents of such a dataset trough the following visualizations:
1. A bar chart depicting the distribution of the messages genre according to 3 categories a) direct, b) news, and c) social.
2. A bar chart depicting the distribution of the messages according to the 36 categories abovementioned.
3. A heatmap depicting the percentage of messages in category `x` also in category `y`.

## Project structure
The project is composed of the 3 components described following.
1. An ETL pipeline that:
	* loads the dataset of real messages,
	* cleans it, and
	* stores it in a SQLite database.
2. A machile learning pipeline that:
	* loads the data from the aforementioned SQLite database,
	* builds a machine learning model, which is trained in the loaded data,
	* and stores the model as a pickle file.
3. A web app that:
	* enables an emergency worker to classify text messages, and
	* depicts the contents of a datasetof real messages trough the three visualizations.

### Folder structure and files
The structure of folders and overall content of the files is described following.
- README.md # this file
- app # contains the files required to run the _Disaster Response_ web app.
	- template # template's folder
		 - master.html # main page of the web app.
		 - go.html # classification result page of the web app.
	- run.py # Python script that runs the web app.

- data # contains the raw and processed dataset of real messages, and the ETL pipeline to process it and stores it a SQLite database.
	- disaster_categories.csv # categories of the dataset of real messages.
	- disaster_messages.cv # dataset of real messages.
	- process_data.py # ETL pipeline to process and store the dataset as a SQLite database.
	- DisasterResponse.db # SQLite database containing the processed dataset of real messages.

- models # contains the serialized machine learning model, and the ML pipeline to generate it.
	- train_classifier.py # ML pipeline to build, train, and sore the model.
	- classifier.pkl # model serialied as pickle file.

## Instructions
### How to run the Python scripts
* Run the following command in the project's root directory to run the ETL pipeline that cleans data and stores in the database:
```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
```
* Run the following command in the project's root directory to run the ML pipeline that trains classifier and saves it:
```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

### How to run the web app
Run the following command in the project's app directory to run the web app:
```
python run.py
```

### How to access the web app
Go to [http://0.0.0.0:3001/](http://0.0.0.0:3001/).

