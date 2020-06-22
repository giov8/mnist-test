from mnist import MNIST 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")
mndata = MNIST("./data")
images, labels = mndata.load_training()

print("Loadind training dataset...")
test_images, test_labels = mndata.load_testing()

clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,), random_state=1)

# Train the first 10000 images
train_x = images
train_y = labels

print("Training model...")
clf.fit(train_x, train_y)

# Test on the next 100 images:
test_x = test_images
expected = test_labels.tolist()

print("Computing predictions...")
predicted = clf.predict(test_x)

print("Acuracy: ", accuracy_score(expected, predicted))
