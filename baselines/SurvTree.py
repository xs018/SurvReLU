import numpy as np

# Column labels.
# These are used only to print the tree.
header = ["color", "diameter", "label"]

class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def is_numeric(self, value):
        """Test if a value is numeric."""
        return isinstance(value, int) or isinstance(value, float)

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if self.is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if self.is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

class SurvTree():
    def __init__(self, criterion, max_depth, min_samples_split=6):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def unique_vals(self, rows, col):
        """Find the unique values for a column in a dataset."""
        return set([row[col] for row in rows])

    def class_counts(self, rows):
        """Counts the number of each type of example in a dataset."""
        counts = {}  # a dictionary of label -> count.
        for row in rows:
            # in our dataset format, the label is always the last column
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts
    
    def partition(self, rows, question:Question):
        """Partitions a dataset.

        For each row in the dataset, check if it matches the question. If
        so, add it to 'true rows', otherwise, add it to 'false rows'.
        """
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows
    
if __name__ == "__main__":
    training_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 3, 'Apple'],
        ['Red', 1, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
    ]

    q = Question(1, 2)

    st = SurvTree('LogRank', 5, 6)

    true_rows, false_rows = st.partition(training_data, Question(0, 'Red'))
    print(true_rows)