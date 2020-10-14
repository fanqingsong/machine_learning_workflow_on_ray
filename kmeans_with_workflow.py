
from csv import reader
from sklearn.cluster import KMeans
import joblib
import ray


ray.init()


# Load a CSV file
def load_csv(filename):
	file = open(filename, "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

def getRawIrisData():
    # Load iris dataset
    filename = 'iris.csv'
    dataset = load_csv(filename)
    print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
    print(dataset[0])
    # convert string columns to float
    for i in range(4):
        str_column_to_float(dataset, i)
    # convert class column to int
    lookup = str_column_to_int(dataset, 4)
    print(dataset[0])
    print(lookup)

    return dataset

@ray.remote
def getTrainData():
    dataset = getRawIrisData()
    trainData = [ [one[0], one[1], one[2], one[3]] for one in dataset ]

    return trainData

@ray.remote
def getNumClusters():
    return 3

@ray.remote
def train(numClusters, trainData):
    print("numClusters=%d" % numClusters)

    model = KMeans(n_clusters=numClusters)

    model.fit(trainData)

    # save model for prediction
    joblib.dump(model, 'model.kmeans')

    return trainData

@ray.remote
def predict(irisData):
    # test saved prediction
    model = joblib.load('model.kmeans')

    # cluster result
    labels = model.predict(irisData)

    print("cluster result")
    print(labels)


def machine_learning_workflow_pipeline():
    trainData = getTrainData.remote()
    numClusters = getNumClusters.remote()
    trainData = train.remote(numClusters, trainData)
    result = predict.remote(trainData)

    result = ray.get(result)
    print("result=", result)



if __name__ == "__main__":
    machine_learning_workflow_pipeline()
