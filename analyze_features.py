import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import argparse
import os

def load_dataset(dataset_path):
    """Load the dataset and separate features from labels."""
    if not os.path.exists(dataset_path):
        print(f"Oops! Dataset not found at: {dataset_path}")
        return None

    df = pd.read_csv(dataset_path)

    if 'damage_label' in df.columns:
        X = df.drop('damage_label', axis=1)
        y = df['damage_label']
        return X, y
    else:
        print("Couldn't find the 'damage_label' column. Please check your dataset.")
        return None

def analyze_feature_importance(model_path, dataset_path, output_dir=None):
    """Check which features the model thinks are most important."""
    model = joblib.load(model_path)
    result = load_dataset(dataset_path)

    if result is None:
        return

    X, y = result

    # Check if model has feature importances (typically tree-based models do)
    classifier = model.named_steps.get('classifier', None)
    if not hasattr(classifier, 'feature_importances_'):
        print("Hmm... This model doesn't support feature importance inspection.")
        return

    importances = classifier.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Add some categorization for easier breakdown
    importance_df['Feature_Type'] = importance_df['Feature'].apply(lambda x: x.split('_')[0])
    importance_df['Feature_Metric'] = importance_df['Feature'].apply(
        lambda x: '_'.join(x.split('_')[1:]) if len(x.split('_')) > 1 else 'unknown'
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

    print("Top 20 Most Important Features According to the Model:")
    for i, (_, row) in enumerate(importance_df.iloc[:20].iterrows()):
        print(f"{i+1}. {row['Feature']}: {row['Importance']:.4f}")

    # Plot top 20
    plt.figure(figsize=(12, 10))
    plt.title('Top 20 Feature Importances', fontsize=16)
    sns.barplot(x='Importance', y='Feature', data=importance_df.iloc[:20])
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_importance_top20.png'))
    else:
        plt.show()

    # Plot by feature type
    type_importance = importance_df.groupby('Feature_Type')['Importance'].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance by Sensor Type', fontsize=16)
    sns.barplot(x=type_importance.index, y=type_importance.values)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_importance_by_type.png'))
    else:
        plt.show()

    # Plot by metric type
    metric_importance = importance_df.groupby('Feature_Metric')['Importance'].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importance by Metric Type', fontsize=16)
    sns.barplot(x=metric_importance.index, y=metric_importance.values)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_importance_by_metric.png'))
    else:
        plt.show()

    return importance_df

def visualize_data_clustering(dataset_path, output_dir=None):
    """Visualize how data clusters using PCA and t-SNE."""
    result = load_dataset(dataset_path)
    if result is None:
        return

    X, y = result

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA Plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Damage Label')
    plt.title('PCA Visualization of Structural Health Data')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.grid(True, alpha=0.3)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'pca_visualization.png'))
    else:
        plt.show()

    # t-SNE Plot
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Damage Label')
    plt.title('t-SNE Visualization of Structural Health Data')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.grid(True, alpha=0.3)

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
    else:
        plt.show()

def analyze_feature_correlations(dataset_path, output_dir=None):
    """Find out which features are most correlated with damage."""
    result = load_dataset(dataset_path)
    if result is None:
        return

    X, y = result
    data = X.copy()
    data['damage_label'] = y

    # Calculate correlation with damage label
    corr_with_label = data.corr()['damage_label'].sort_values(ascending=False)

    print("Top 20 Features Most Correlated with Damage:")
    print(corr_with_label.iloc[:21])

    plt.figure(figsize=(12, 10))
    plt.title('Correlation with Damage Label', fontsize=16)
    sns.barplot(x=corr_with_label.iloc[1:21].values, y=corr_with_label.iloc[1:21].index)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'feature_correlation_with_damage.png'))
    else:
        plt.show()

    # Correlation heatmap for top features
    top_features = list(corr_with_label.iloc[1:11].index)
    top_features.append('damage_label')

    corr_matrix = data[top_features].corr()

    plt.figure(figsize=(12, 10))
    plt.title('Correlation Matrix (Top 10 Features)', fontsize=16)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'correlation_matrix_top10.png'))
    else:
        plt.show()

def analyze_class_separability(dataset_path, output_dir=None):
    """Check how well top features separate damaged from undamaged samples."""
    result = load_dataset(dataset_path)
    if result is None:
        return

    X, y = result
    data = X.copy()
    data['damage_label'] = y

    corr_with_label = data.corr()['damage_label'].sort_values(ascending=False)
    top_features = list(corr_with_label.iloc[1:5].index)

    # Pairplot for visual separation
    plt.figure(figsize=(14, 10))
    sns.pairplot(data=data, vars=top_features, hue='damage_label', palette='viridis')
    plt.suptitle('Pair Plot of Top 4 Features', y=1.02, fontsize=16)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'pairplot_top4.png'))
    else:
        plt.show()

    # Histograms for distribution
    plt.figure(figsize=(14, 10))
    for i, feature in enumerate(top_features):
        plt.subplot(2, 2, i+1)
        sns.histplot(data=data, x=feature, hue='damage_label', kde=True, palette='viridis')
        plt.title(f'Distribution: {feature}')

    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    else:
        plt.show()

def main():
    """Parse command-line arguments and run the selected analysis tasks."""
    parser = argparse.ArgumentParser(description="Structural Health Monitoring - Feature Analysis")
    parser.add_argument("--model", type=str, default="structural_damage_model.pkl",
                        help="Path to your trained model (Pickle format)")
    parser.add_argument("--dataset", type=str, default="processed_dataset.csv",
                        help="CSV file with processed features and labels")
    parser.add_argument("--output", type=str, default="analysis_results",
                        help="Folder to save plots and results")
    parser.add_argument("--importance", action="store_true", help="Run feature importance analysis")
    parser.add_argument("--clustering", action="store_true", help="Visualize clustering (PCA & t-SNE)")
    parser.add_argument("--correlations", action="store_true", help="Check feature correlations")
    parser.add_argument("--separability", action="store_true", help="Visualize class separability")
    parser.add_argument("--all", action="store_true", help="Run all available analyses")

    args = parser.parse_args()

    if args.all or args.importance:
        print("\n🔍 Running Feature Importance Analysis...")
        analyze_feature_importance(args.model, args.dataset, args.output)

    if args.all or args.clustering:
        print("\n🔍 Visualizing Clustering in Data...")
        visualize_data_clustering(args.dataset, args.output)

    if args.all or args.correlations:
        print("\n📈 Analyzing Feature Correlations...")
        analyze_feature_correlations(args.dataset, args.output)

    if args.all or args.separability:
        print("\n📊 Exploring Class Separability...")
        analyze_class_separability(args.dataset, args.output)

    print("\n✅ All selected analyses are complete!")

if __name__ == "__main__":
    main()
