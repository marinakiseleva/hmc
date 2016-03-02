hmc
===

Decision Tree Hierachical Multi-Classifier
------------------------------------------
A thin wrapper for sklearn DecisionTreeClassifier for hierarchical classes, implementing HSC-Clus [1].

Define a class hierarchy. Class hierarchies are constructed with the root node. Nodes are added with child parent pairs.
```python
import hmc
ch = hmc.ClassHierarchy("colors")
ch.add_node("light", "colors")
ch.add_node("dark", "colors")
ch.add_node("white", "light")
ch.add_node("black", "dark")
ch.add_node("gray", "dark")
ch.add_node("slate", "gray")
ch.add_node("ash", "gray")
```
Pretty print it.
```python
ch.print_()
```
```
└─colors
  ├─dark
  │ ├─black
  │ └─gray
  │   ├─ash
  │   └─slate
  └─light
    └─white
```

Load some data from the included functions and split for training. The class hierarchy itself can also be loaded from the module.
```python
ch = hmc.load_shades_class_hierachy()
X, y = hmc.load_shades_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50)
```

Each parent node, of at least one child, will generate a decision tree classification stage. Stages are assigned depth first, ascending alpha.
```python
dt.stages
```
```
[{'classes': ['dark', 'light', 'colors'],
  'depth': 0,
  'labels': ['dark', 'light'],
  'stage': 'colors'},
{'classes': ['black', 'gray', 'dark'],
  'depth': 1, 'labels': ['black', 'gray'],
  'stage': 'dark'},
{'classes': ['ash', 'slate', 'gray'],
  'depth': 2, 'labels': ['ash', 'slate'],
  'stage': 'gray'},
{'classes': ['white', 'light'],
  'depth': 1, 'labels': ['white'],
  'stage': 'light'}]
```
The hmc.DecisionTreeHierarchicalClassifier is idiomatic to the sklearn tree.DecisionTreeClassifier. Fit, predict and score the same way. Traditional multi-classification accuracy is comparable.
```python
from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)
dt_predicted = dt.predict(X_test)
dt_accuracy = dt.score(X_test, y_test)

dth = hmc.DecisionTreeHierarchicalClassifier(ch)
dth = dth.fit(X_train, y_train)
dth_predicted = dth.predict(X_test)
dth_accuracy = dth.score(X_test, y_test)
```
```
>>> dt_accuracy
0.46561886051080548
>>> dth_accuracy
0.46758349705304519
```
Hierarchically adjusted classification accuracy scoring is available in addition to traditional accuracy. This metric averages accuracy at each classification stage, penalizing the least harshly cases of the mis-classification of sibling nodes, and most harshly cases where true and predicted classes share no ancestors in the hierarchy.
```python
dth_accuracy_adjusted = dth.score_adjusted(X_test, y_test)
```
```
>>> dth_accuracy_adjusted
0.66115923150295042
```

1. Vens, C., Struyf, J., Schietgat, L., Džeroski, S., & Blockeel, H. (2008). Decision trees for hierarchical multi-label classification. Mach Learn Machine Learning, 73(2), 185-214.
