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
>>> ch.print_()
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
>>> dt.stages
[{'classes': ['dark', 'light', 'colors'],
  'depth': 0,
  'labels': ['dark', 'light'],
  'stage': 'colors'},
{'classes': ['black', 'gray', 'dark'],
  'depth': 1,
  'labels': ['black', 'gray'],
  'stage': 'dark'},
{'classes': ['ash', 'slate', 'gray'],
  'depth': 2,
  'labels': ['ash', 'slate'],
  'stage': 'gray'},
{'classes': ['white', 'light'],
  'depth': 1,
  'labels': ['white'],
  'stage': 'light'}]
```
The hmc.DecisionTreeHierarchicalClassifier is idiomatic to the sklearn tree.DecisionTreeClassifier. Fit, predict and score the same way. Traditional multi-classification average accuracy is comparable.
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
```python
>>> dt_accuracy
0.4400785854616896
>>> dth_accuracy
0.46561886051080548
```
Additional hierarchical multi-classification specific metrics [2] are provided.
```python
import hmc.metrics as metrics

>>> metrics.accuracy_score(ch, dth_predicted, y_test)
0.46561886051080548
>>> metrics.precision_score_ancestors(ch, dth_predicted, y_test)
0.8108614232209738
>>> metrics.recall_score_ancestors(ch, dth_predicted, y_test)
0.7988929889298892
>>> metrics.f1_score_ancestors(ch, dth_predicted, y_test)
0.8048327137546468
>>> metrics.precision_score_descendants(ch, dth_predicted, y_test)
0.6160337552742616
>>> metrics.recall_score_descendants(ch, dth_predicted, y_test)
0.6576576576576577
>>> metrics.f1_score_descendants(ch, dth_predicted, y_test)
0.636165577342048
```
Ancestor and Descendant precision and recall scores are calculated as the fraction of shared ancestor or descendant classes over the sum of either the predicted or true class for precision and recall respectively [3].
```python
true = ['dark', 'white', 'gray']

pred_sibling = ['dark', 'white', 'black']

>>> metrics.accuracy_score(ch, pred_sibling, true)
0.66666666666666663
>>> metrics.precision_score_ancestors(ch, pred_sibling, true)
0.8
>>> metrics.precision_score_descendants(ch, pred_sibling, true)
0.6666666666666666

pred_narrower = ['dark', 'white', 'ash']

>>> metrics.accuracy_score(ch, pred_narrower, true)
0.66666666666666663
>>> metrics.precision_score_ancestors(ch, pred_narrower, true)
1.0
>>> metrics.precision_score_descendants(ch, pred_narrower, true)
0.7777777777777778

pred_broader = ['dark', 'white', 'dark']

>>> metrics.accuracy_score(ch, pred_broader, true)
0.66666666666666663
>>> metrics.precision_score_ancestors(ch, pred_broader, true)
0.8
>>> metrics.precision_score_descendants(ch, pred_broader, true)
1.0
```

1. Vens, C., Struyf, J., Schietgat, L., Džeroski, S., & Blockeel, H. (2008). Decision trees for hierarchical multi-label classification. Mach Learn Machine Learning, 73(2), 185-214.
2. Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information Processing & Management, 45(4), 427-437. doi:10.1016/j.ipm.2009.03.002
3. Costa, E., Lorena, A., Carvalho, A., & Freitas, A. (2007). A review of performance evaluation measures for hierarchical classifiers. In Proceedings of the AAAI
2007 workshop "Evaluation methods for machine learning" (pp. 1–6).
