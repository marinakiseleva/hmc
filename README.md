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
0.48526522593320237
>>> dth_accuracy
0.45776031434184677
```
Additional hierarchical multi-classification specific metrics [2] are provided.
```python
import hmc.metrics as metrics

>>> metrics.accuracy_score(ch, y_test, dth_predicted)
0.45776031434184677
>>> metrics.precision_score_ancestors(ch, y_test, dth_predicted)
0.8
>>> metrics.recall_score_ancestors(ch, y_test, dth_predicted)
0.8052190121155638
>>> metrics.f1_score_ancestors(ch, y_test, dth_predicted)
0.8026010218300047
>>> metrics.precision_score_descendants(ch, y_test, dth_predicted)
0.647191011235955
>>> metrics.recall_score_descendants(ch, y_test, dth_predicted)
0.6260869565217392
>>> metrics.f1_score_descendants(ch, y_test, dth_predicted)
0.63646408839779
```
Ancestor and Descendant precision and recall scores are calculated as the fraction of shared ancestor or descendant classes over the sum of either the predicted or true class for precision and recall respectively [3].
```python
true = ['dark', 'white', 'gray']

pred_sibling = ['dark', 'white', 'black']

>>> metrics.accuracy_score(ch, true, pred_sibling)
0.66666666666666663
>>> metrics.precision_score_ancestors(ch, true, pred_sibling)
0.8
>>> metrics.precision_score_descendants(ch, true, pred_sibling)
0.8571428571428571

pred_narrower = ['dark', 'white', 'ash']

>>> metrics.accuracy_score(ch, true, pred_narrower)
0.66666666666666663
>>> metrics.precision_score_ancestors(ch, true, pred_narrower)
0.8333333333333334
>>> metrics.precision_score_descendants(ch, true, pred_narrower)
1.0

pred_broader = ['dark', 'white', 'dark']

>>> metrics.accuracy_score(ch, true, pred_broader)
0.66666666666666663
>>> metrics.precision_score_ancestors(ch, true, pred_broader)
1.0
>>> metrics.precision_score_descendants(ch, true, pred_broader)
0.8181818181818182
```

1. Vens, C., Struyf, J., Schietgat, L., Džeroski, S., & Blockeel, H. (2008). Decision trees for hierarchical multi-label classification. Mach Learn Machine Learning, 73(2), 185-214.
2. Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information Processing & Management, 45(4), 427-437. doi:10.1016/j.ipm.2009.03.002
3. Costa, E., Lorena, A., Carvalho, A., & Freitas, A. (2007). A review of performance evaluation measures for hierarchical classifiers. In Proceedings of the AAAI
2007 workshop "Evaluation methods for machine learning" (pp. 1–6).
