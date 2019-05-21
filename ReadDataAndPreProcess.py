from sklearn import model_selection,preprocessing

def ReadDataAndPreProcess(file,test_size):
    # load the dataset
    data = open(file).read()
    AllSamplesX, AllSamplesY = [], []
    for line in data.split("\n"):
        content = line.split()
        AllSamplesY.append(content[0])
        AllSamplesX.append(" ".join(content[1:]))

    TrainX,TestX,TrainY,TestY = model_selection.train_test_split(AllSamplesX,AllSamplesY,test_size=test_size)

    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    encoder.fit(AllSamplesY)
    TrainY = encoder.transform(TrainY)
    TestY = encoder.transform(TestY)

    return AllSamplesX,AllSamplesY, TrainX,TrainY,TestX,TestY