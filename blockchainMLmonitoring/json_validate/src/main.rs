use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

// Define a complex data structure for JSON validation
#[derive(Serialize, Deserialize, Debug, Clone)]
struct ComplexData {
    id: u64,
    name: String,
    features: Vec<f64>,
    labels: Option<Vec<String>>,
    nested: HashMap<String, NestedData>,
    timestamp: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct NestedData {
    value: f64,
    category: String,
    sub_features: Vec<f64>,
}

// Error handling with custom error type
#[derive(Debug)]
enum ValidationError {
    Io(io::Error),
    Json(serde_json::Error),
}

impl From<io::Error> for ValidationError {
    fn from(err: io::Error) -> Self {
        ValidationError::Io(err)
    }
}

impl From<serde_json::Error> for ValidationError {
    fn from(err: serde_json::Error) -> Self {
        ValidationError::Json(err)
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ValidationError::Io(e) => write!(f, "IO Error: {}", e),
            ValidationError::Json(e) => write!(f, "JSON Error: {}", e),
        }
    }
}

impl Error for ValidationError {}

// Function to read and parse JSON file
fn read_json_file<P: AsRef<Path>>(path: P) -> Result<Vec<ComplexData>, ValidationError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut data: Vec<ComplexData> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            let item: ComplexData = serde_json::from_str(&line)?;
            data.push(item);
        }
    }
    Ok(data)
}

// Simple feature extraction
fn extract_features(data: &[ComplexData]) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut features_vec: Vec<Vec<f64>> = Vec::new();
    let mut labels_vec: Vec<usize> = Vec::new();

    for item in data {
        let mut feat = item.features.clone();
        for (_, nested) in &item.nested {
            feat.extend_from_slice(&nested.sub_features);
            feat.push(nested.value);
        }
        features_vec.push(feat);

        // Simple label encoding
        if let Some(labels) = &item.labels {
            let hash = labels
                .iter()
                .fold(0usize, |acc, l| acc.wrapping_add(l.len()));
            labels_vec.push(hash % 10);
        } else {
            labels_vec.push(0);
        }
    }

    (features_vec, labels_vec)
}

// Simple normalization
fn normalize_features(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if features.is_empty() {
        return Vec::new();
    }

    let num_features = features[0].len();
    let mut normalized = Vec::with_capacity(features.len());

    // Calculate means and stds for each feature
    let mut means = vec![0.0; num_features];
    let mut stds = vec![0.0; num_features];

    for feature_idx in 0..num_features {
        let values: Vec<f64> = features.iter().map(|f| f[feature_idx]).collect();

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();

        means[feature_idx] = mean;
        stds[feature_idx] = if std > 0.0 { std } else { 1.0 };
    }

    // Normalize each feature vector
    for feature_vec in features {
        let mut normalized_vec = Vec::with_capacity(num_features);
        for (i, &value) in feature_vec.iter().enumerate() {
            normalized_vec.push((value - means[i]) / stds[i]);
        }
        normalized.push(normalized_vec);
    }

    normalized
}

// Simple KNN classifier
struct SimpleKNN {
    training_features: Vec<Vec<f64>>,
    training_labels: Vec<usize>,
    k: usize,
}

impl SimpleKNN {
    fn new(k: usize) -> Self {
        Self {
            training_features: Vec::new(),
            training_labels: Vec::new(),
            k,
        }
    }

    fn fit(&mut self, features: Vec<Vec<f64>>, labels: Vec<usize>) {
        self.training_features = features;
        self.training_labels = labels;
    }

    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn predict(&self, features: &[Vec<f64>]) -> Vec<usize> {
        features
            .iter()
            .map(|feature| {
                // Calculate distances to all training points
                let mut distances: Vec<(usize, f64)> = self
                    .training_features
                    .iter()
                    .enumerate()
                    .map(|(idx, train_feat)| (idx, self.euclidean_distance(feature, train_feat)))
                    .collect();

                // Sort by distance and take k nearest
                distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                // Count votes from k nearest neighbors
                let mut votes = HashMap::new();
                for &(idx, _) in distances.iter().take(self.k) {
                    let label = self.training_labels[idx];
                    *votes.entry(label).or_insert(0) += 1;
                }

                // Return the label with most votes
                votes
                    .into_iter()
                    .max_by_key(|&(_, count)| count)
                    .map(|(label, _)| label)
                    .unwrap_or(0)
            })
            .collect()
    }
}

// Simple decision tree (very basic implementation)
struct SimpleDecisionTree {
    threshold: f64,
    feature_index: usize,
    left_label: usize,
    right_label: usize,
}

impl SimpleDecisionTree {
    fn new() -> Self {
        Self {
            threshold: 0.0,
            feature_index: 0,
            left_label: 0,
            right_label: 0,
        }
    }

    fn fit(&mut self, features: &[Vec<f64>], labels: &[usize]) {
        if features.is_empty() {
            return;
        }

        // Very simple heuristic: find the feature and threshold that best separates the data
        let num_features = features[0].len();
        let mut best_impurity = f64::MAX;

        for feature_idx in 0..num_features {
            let values: Vec<f64> = features.iter().map(|f| f[feature_idx]).collect();
            let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            if min_val >= max_val {
                continue;
            }

            // Try a few thresholds
            for step in 1..10 {
                let threshold = min_val + (max_val - min_val) * (step as f64) / 10.0;
                let impurity = self.calculate_impurity(features, labels, feature_idx, threshold);

                if impurity < best_impurity {
                    best_impurity = impurity;
                    self.threshold = threshold;
                    self.feature_index = feature_idx;
                }
            }
        }

        // Determine labels for left and right branches manually
        let mut left_labels = Vec::new();
        let mut right_labels = Vec::new();

        for (feature, &label) in features.iter().zip(labels.iter()) {
            if feature[self.feature_index] <= self.threshold {
                left_labels.push(label);
            } else {
                right_labels.push(label);
            }
        }

        self.left_label = Self::majority_vote(&left_labels);
        self.right_label = Self::majority_vote(&right_labels);
    }

    fn calculate_impurity(
        &self,
        features: &[Vec<f64>],
        labels: &[usize],
        feature_idx: usize,
        threshold: f64,
    ) -> f64 {
        let mut left_labels = Vec::new();
        let mut right_labels = Vec::new();

        for (feature, &label) in features.iter().zip(labels.iter()) {
            if feature[feature_idx] <= threshold {
                left_labels.push(label);
            } else {
                right_labels.push(label);
            }
        }

        let left_gini = Self::gini_impurity(&left_labels);
        let right_gini = Self::gini_impurity(&right_labels);

        let left_weight = left_labels.len() as f64 / labels.len() as f64;
        let right_weight = right_labels.len() as f64 / labels.len() as f64;

        left_weight * left_gini + right_weight * right_gini
    }

    fn gini_impurity(labels: &[usize]) -> f64 {
        if labels.is_empty() {
            return 0.0;
        }

        let mut counts = HashMap::new();
        for &label in labels {
            *counts.entry(label).or_insert(0) += 1;
        }

        let total = labels.len() as f64;
        counts
            .values()
            .map(|&count| {
                let prob = count as f64 / total;
                prob * (1.0 - prob)
            })
            .sum()
    }

    fn majority_vote(labels: &[usize]) -> usize {
        let mut counts = HashMap::new();
        for &label in labels {
            *counts.entry(label).or_insert(0) += 1;
        }
        counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label)
            .unwrap_or(0)
    }

    fn predict(&self, features: &[Vec<f64>]) -> Vec<usize> {
        features
            .iter()
            .map(|feature| {
                if feature[self.feature_index] <= self.threshold {
                    self.left_label
                } else {
                    self.right_label
                }
            })
            .collect()
    }
}

// Simple validation using ensemble
fn validate_data(
    train_features: &[Vec<f64>],
    train_labels: &[usize],
    val_features: &[Vec<f64>],
) -> Vec<bool> {
    // Train multiple simple models
    let mut knn = SimpleKNN::new(3);
    knn.fit(train_features.to_vec(), train_labels.to_vec());

    let mut tree = SimpleDecisionTree::new();
    tree.fit(train_features, train_labels);

    // Get predictions from both models
    let knn_pred = knn.predict(val_features);
    let tree_pred = tree.predict(val_features);

    // Ensemble: if both models agree, use their prediction
    knn_pred
        .iter()
        .zip(tree_pred.iter())
        .map(|(&knn, &tree)| knn == tree && knn != 0)
        .collect()
}

// Simple anomaly detection using distance-based approach
fn detect_anomalies(features: &[Vec<f64>], threshold_percentile: f64) -> Vec<bool> {
    if features.len() < 2 {
        return vec![false; features.len()];
    }

    let mut distances = Vec::new();

    // Calculate distances to nearest neighbor for each point
    for (i, feature) in features.iter().enumerate() {
        let mut min_distance = f64::INFINITY;

        for (j, other_feature) in features.iter().enumerate() {
            if i != j {
                let distance = euclidean_distance(feature, other_feature);
                if distance < min_distance {
                    min_distance = distance;
                }
            }
        }

        distances.push(min_distance);
    }

    // Find threshold
    let mut sorted_distances = distances.clone();
    sorted_distances.sort_by(|a, b| a.partial_cmp(&b).unwrap());

    let threshold_index = (sorted_distances.len() as f64 * threshold_percentile) as usize;
    let threshold = sorted_distances[threshold_index.min(sorted_distances.len() - 1)];

    // Mark points with large distances as anomalies
    distances.iter().map(|&d| d <= threshold).collect()
}

// Euclidean distance helper function
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn main() -> Result<(), ValidationError> {
    // Read and parse data
    let train_data = read_json_file("train.jsonl")?;
    let validation_data = read_json_file("validate.jsonl")?;

    // Extract features
    let (train_features, train_labels) = extract_features(&train_data);
    let (val_features, _) = extract_features(&validation_data);

    // Normalize features
    let normalized_train_features = normalize_features(&train_features);
    let normalized_val_features = normalize_features(&val_features);

    println!(
        "Processed {} training samples",
        normalized_train_features.len()
    );
    println!(
        "Processed {} validation samples",
        normalized_val_features.len()
    );

    // Validate data using ensemble
    let validations = validate_data(
        &normalized_train_features,
        &train_labels,
        &normalized_val_features,
    );

    // Detect anomalies
    let anomalies = detect_anomalies(&normalized_val_features, 0.95);

    // Print results
    for (i, (valid, anomaly)) in validations.iter().zip(anomalies.iter()).enumerate() {
        println!("Data point {}: Valid={}, Anomaly={}", i, valid, !*anomaly);
    }

    // Calculate statistics
    let valid_count = validations.iter().filter(|&&v| v).count();
    let anomaly_count = anomalies.iter().filter(|&&a| !a).count();

    println!("\nSummary:");
    println!("Valid data points: {}/{}", valid_count, validations.len());
    println!("Anomalies detected: {}/{}", anomaly_count, anomalies.len());

    Ok(())
}
