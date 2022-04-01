import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_feat_chart(feat, label):
    """Visualize HAR data"""
    fig, axs = plt.subplots(nrows=3, figsize=(15,10))
    [ax.grid(True) for ax in axs]
    
    ta_df = feat[["total_acc_x", "total_acc_y", "total_acc_z"]]
    ta_df = ta_df.to_frame("value").reset_index().rename({"level_0": "axis", "level_1": "step"}, axis=1)

    sns.lineplot(x="step", y="value", hue="axis", data=ta_df, ax=axs[0]).set_title(f"Total acceleration (label={label})")
    
    # body acc
    ba_df = feat[["body_acc_x", "body_acc_y", "body_acc_z"]]
    ba_df = ba_df.to_frame("value").reset_index().rename({"level_0": "axis", "level_1": "step"}, axis=1)

    sns.lineplot(x="step", y="value", hue="axis", data=ba_df, ax=axs[1]).set_title(f"Body acceleration (label={label})")
    
    # body gyro
    bg_df = feat[["body_gyro_x", "body_gyro_y", "body_gyro_z"]]
    bg_df = bg_df.to_frame("value").reset_index().rename({"level_0": "axis", "level_1": "step"}, axis=1)

    sns.lineplot(x="step", y="value", hue="axis", data=bg_df, ax=axs[2]).set_title(f"Body gyro (label={label})")
    
    return fig

def plot_feat_tensor_chart(feat_tensor, label, height=400, width=1000):
    """Visual HAR data from one tensor"""
    feat_arr = feat_tensor.numpy()
    feat_cols = ["total_acc_x", "total_acc_y", "total_acc_z", "body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"]
    df = pd.DataFrame(feat_arr, columns=feat_cols)
    chart_df = df.reset_index().melt("index")
    chart_df.rename({"index":"step", "variable":"feat"}, axis=1, inplace=True)
    
    chart = alt.Chart(chart_df).mark_line(point=False).encode(
            x="step",
            y=alt.Y("value"),
            tooltip=["step", "value"],
            color="feat"
        ).properties(
            title={
                "text": f"Sample data (label={label})",
            },
            height=height,
            width=width,
        ).interactive()
    return chart


def plot_conf_matrix_chart(true_labels, preds, height=400, width=400):
    conf_mat = confusion_matrix(true_labels, preds)

    X = []
    Y = []
    Z = []
    labels = np.unique(true_labels)

    for i in labels:
        for j in labels:
            X.append(i)
            Y.append(j)
            Z.append(conf_mat[i][j])
    chart_df = pd.DataFrame({"true": X, "pred": Y, "n": Z})

    chart = alt.Chart(chart_df).mark_rect().encode(
        x="pred:O",
        y="true:O",
        color="n:Q",
        tooltip=["true", "pred", "n"],
    ).properties(
        title={
            "text": "Confusion matrix",
        },
        height=height,
        width=width,
    ).interactive()
    
    return chart