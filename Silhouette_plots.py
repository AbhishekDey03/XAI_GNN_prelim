from sklearn.metrics import silhouette_samples
import numpy as np
import matplotlib.pyplot as plt

def plot_silhouette(features: np.ndarray,
                    labels: np.ndarray,
                    title: str,
                    save: bool = False,
                    label_to_color: dict = None,
                    label_names: dict = None,
                    **kwargs):

    sil_vals = silhouette_samples(features, labels)
    avg_sil = sil_vals.mean()

    n_classes = len(np.unique(labels))
    y_lower = 0
    fig, ax = plt.subplots(figsize=(8, 6))

    for cls in np.unique(labels):
        cls_sil = np.sort(sil_vals[labels == cls])
        size = cls_sil.shape[0]
        y_upper = y_lower + size

        color = label_to_color.get(cls, 'gray') if label_to_color else 'gray'
        label_str = label_names.get(cls, str(cls)) if label_names else str(cls)

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cls_sil,
                         alpha=0.7, label=label_str, color=color)
        y_lower = y_upper

    plot_name = kwargs.get('plot_name', None)
    ax.axvline(avg_sil, color='red', linestyle='--', label=f'Avg = {avg_sil:.2f}')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Silhouette Width')
    ax.set_xlim([-0.4, 1.0])
    ax.set_ylabel('Class Label')
    ax.set_yticks([])
    ax.legend(loc='upper right')
    plt.savefig(plot_name) if save and plot_name else None
    plt.tight_layout()
    plt.show()
