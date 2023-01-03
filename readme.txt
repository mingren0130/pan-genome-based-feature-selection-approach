Python packge

	tqdm
	numpy
	pandas
	sklearn
	xgboost (requires cmake 3.1.3 or higher as of 6/25/2021)

Program Usage

	python3 SVM.py 	-i <input filename (please specify .csv file format)>
			        -o <feature output filename>

Datasets
	sample data files were provided to test the efficacy of the CVFS program. Please unzip the CSV files using the following command:

Example command:
	(Simple run using default command; assume running on the ampicillin.csv dataset)
	$ python3 SVM.py -i ampicillin.csv -o ampicillin.csv

Contributing authors
Ming-Ren Yang 
Graduate Institute of Biomedical Informatics, Taipei Medical University, Taipei, Taiwan

Version
0.1
