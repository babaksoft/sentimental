import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .config import config


def feature_target_split(csv_path):
    data = pd.read_csv(csv_path)
    x = data.drop(config.TARGET, axis=1)
    y = data[config.TARGET]
    return x, y


# Plot confusion matrix using a default matplotlib colormap
def plot_confusion_matrix(pipeline, x, y, normalize=None):
    cmap = "summer"
    y_predict = pipeline.predict(x)
    clf = pipeline.named_steps["classifier"]
    cm = confusion_matrix(
        y, y_predict, labels=clf.classes_, normalize=normalize
    )
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=clf.classes_
    )

    cm_display.plot(cmap=cmap)
    path = config.METRICS_DIR / f"cm_baseline.png"
    plt.savefig(path)
    plt.close()
