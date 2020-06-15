from mnist import MNIST 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")
mndata = MNIST("./data")
images, labels = mndata.load_training()

clf = RandomForestClassifier(n_estimators=100)

# Train the first 10000 images
train_x = images[:10000]
train_y = labels[:10000]

print("Training model...")
clf.fit(train_x, train_y)

# Test on the next 100 images:
test_x = images[10000:10100]
expected = labels[10000:10100].tolist()

print("Computing predictions...")
predicted = clf.predict(test_x)

print("Acuracy: ", accuracy_score(expected, predicted))