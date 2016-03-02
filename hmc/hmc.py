"""
The `hmc` module is a decision tree based model for hierachical multi-classification.
"""

from __future__ import print_function
from __future__ import division

from sklearn import tree

import numpy as np
import pandas as pd

__all__ = ["ClassHierarchy", "DecisionTreeHierarchicalClassifier"]

# =============================================================================
# Class Hierarchy
# =============================================================================

class ClassHierarchy:
    """
    Class for class heirarchy.

    Parameters
    ----------
        root :

    Attributes
    ----------

    """
    def __init__(self, root):
        self.root = root
        self.nodes = {}

    def _get_parent(self, child):
        # Return the parent of this node
        return self.nodes[child] if child != self.root else self.root

    def _get_children(self, parent):
        # Return a list of children nodes in alpha order
        return sorted([child for child, childs_parent in self.nodes.iteritems() if childs_parent == parent])

    def _is_descendant(self, parent, child):
        while child != self.class_hierarchy.root and child != parent:
            child = self.class_hierarchy._get_parent(child)
        return child == parent

    def _is_ancestor(self, parent, child):
        return _is_descendant(parent, child)

    def _depth_first_print(self, parent, indent, last):
        print(indent, end="")
        if last:
            print(u"\u2514\u2500", end="")
            indent += "  "
        else:
            print(u"\u251C\u2500", end="")
            indent += u"\u2502 "
        print(parent)
        num_nodes = len(self._get_children(parent))
        node_count = 0
        for node in self._get_children(parent):
            node_count += 1
            self._depth_first_print(node, indent, node_count == num_nodes)

    def _depth_first(self, parent, classes):
        classes.append(parent)
        for node in self._get_children(parent):
            self._depth_first(node, classes)

    def add_node(self, child, parent):
        """
        Add a child-parent node to the class hierarchy.
        """
        self.nodes[child] = parent

    def nodes_(self):
        """
        Return the hierarchy classes as a list of child-parent nodes.
        """
        return self.nodes

    def classes_(self):
        """
        Return the hierarchy classes as a list of unique classes.
        """
        classes = []
        self._depth_first(self.root, classes)
        return classes

    def print_(self):
        """
        Pretty print the class hierarchy.
        """
        self._depth_first_print(self.root, "", True)

# =============================================================================
# Decision Tree Hierarchical Classifier
# =============================================================================

class DecisionTreeHierarchicalClassifier:

    def __init__(self, class_hierarchy):
        self.stages = []
        self.class_hierarchy = class_hierarchy
        self._depth_first_stages(self.stages, self.class_hierarchy.root, 0)

    def _depth_first_class_prob(self, tree, node, indent, last, hand):
        if node == -1:
            return
        print(indent, end="")
        if last:
            print(u"\u2514\u2500", end="")
            indent += "    "
        else:
            print(u"\u251C\u2500", end="")
            indent += u"\u2502   "
        print(hand + " " + str(node))
        for k, count in enumerate(tree.tree_.value[node][0]):
            print(indent + str(tree.classes_[k]) + ":" + str(stage(count / tree.tree_.n_node_samples[node], 2)))
        self._depth_first_class_prob(tree, tree.tree_.children_right[node], indent, False, "R")
        self._depth_first_class_prob(tree, tree.tree_.children_left[node], indent, True, "L")

    def _depth_first_stages(self, stages, parent, depth):
        # Get the children of this parent
        children = self.class_hierarchy._get_children(parent)
        # If there are children, build a classification stage
        if len(children) > 0:
            # Assign stage props and append
            stage = {}
            stage['depth'] = depth
            stage['stage'] = parent
            stage['labels'] = children
            stage['classes'] = stage['labels'] + [stage['stage']]
            stage['target'] = 'target_stage_' + parent
            stages.append(stage)
            # Recurse through children
            for node in children:
                self._depth_first_stages(stages, node, depth + 1)

    def _recode_label(self, classes, label):
        # Reassign labels to their parents until either we hit the root, or an output class
        while label != self.class_hierarchy.root and label not in classes:
            label = self.class_hierarchy._get_parent(label)
        return label

    def _prep_data(self, X, y):
        # Design matrix columns
        dm_cols = range(0, X.shape[1])
        # Target columns
        target = X.shape[1]
        # Dataframe
        df = pd.concat([X, y], axis=1, ignore_index=True)
        # Create a target column for each stage with the recoded labels
        for stage_number, stage in enumerate(self.stages):
            df[stage['target']] = pd.DataFrame.apply(
                df[[target]],
                lambda row: self._recode_label(stage['classes'],
                row[target]),
                axis=1)
        return df, dm_cols

    def fit(self, X, y):
        """
        Build a decision tree multi-classifier from training data (X, y).
        """
        # Prep data
        df, dm_cols = self._prep_data(X, y)
        # Fit each stage
        for stage_number, stage in enumerate(self.stages):
            stage['tree'] = tree.DecisionTreeClassifier()
            stage['tree'] = stage['tree'].fit(
                df[df[stage['target']].isin(stage['classes'])][dm_cols],
                df[df[stage['target']].isin(stage['classes'])][[stage['target']]])
        return self

    def _predict_stages(self, X):
        # Score each stage
        for stage_number, stage in enumerate(self.stages):
            if stage_number == 0:
                y_hat = pd.DataFrame([self.class_hierarchy.root] * len(X), columns=[self.class_hierarchy.root], index=X.index)
            else:
                y_hat[stage['stage']] = y_hat[self.stages[stage_number - 1]['stage']]
            dm = X[y_hat[stage['stage']].isin([stage['stage']])]
            # combine_first reorders DataFrames, so we have to do this the ugly way
            y_hat_stage = pd.DataFrame(stage['tree'].predict(dm), index=dm.index)
            y_hat = y_hat.assign(stage_col=y_hat_stage)
            y_hat.stage_col = y_hat.stage_col.fillna(y_hat[stage['stage']])
            y_hat = y_hat.drop(stage['stage'], axis=1)
            y_hat = y_hat.rename(columns={'stage_col': stage['stage']})
        # Return predicted class for each stage
        return y_hat

    def predict(self, X):
        """
        Predict class for X.
        """
        y_hat = self._predict_stages(X)
        # Return only final predicted class
        return y_hat.ix[:, y_hat.shape[1] - 1].as_matrix()

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data (X, y).
        """
        classes = pd.DataFrame(self.predict(X), columns=['y_hat'], index=y.index)
        classes['y'] = pd.DataFrame(y)
        classes['correct'] = classes.apply(lambda row: 1 if row['y_hat'] == row['y'] else 0, axis=1)
        return classes[['correct']].mean()[0]

    def _score_stages(self, X, y):
        y_hat = self._predict_stages(X)
        y = pd.DataFrame(y)
        y_classes = pd.DataFrame(index=y.index)

        def assign_ancestor(classes, descendent):
            while descendent not in classes and descendent != self.class_hierarchy.root:
                descendent = self.class_hierarchy._get_parent(descendent)
            if descendent == self.class_hierarchy.root and self.class_hierarchy.root not in classes:
              descendent = ""
            return descendent

        accuracies = []
        for stage in self.stages:
            y_hat[stage['stage'] + "_true"] = y.apply(lambda row: assign_ancestor(stage['classes'], row[0]), axis=1)
            y_hat[stage['stage'] + "_correct"] = y_hat.apply(lambda row: 1 if row[stage['stage'] + "_true"] == row[stage['stage']] else 0, axis=1)
            y_hat[stage['stage'] + "_included"] = y_hat.apply(lambda row: 1 if len(row[stage['stage'] + "_true"]) > 0 else 0, axis=1)
            accuracy = y_hat[[stage['stage'] + "_correct"]].sum()[0] / y_hat[[stage['stage'] + "_included"]].sum()[0]
            accuracies.append(accuracy)
        return accuracies

    def score_adjusted(self, X, y):
        """
        Returns the hierachy adjusted mean accuracy on the given test data (X, y).
        """
        accuracies = self._score_stages(X, y)
        return (1 / len(self.stages)) * sum(accuracies)
