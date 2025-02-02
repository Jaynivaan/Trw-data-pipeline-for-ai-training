// ================================
// Dependencies and Imports
// ================================
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::collections::HashMap;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio::time::{sleep, Duration};
use tokio::sync::Mutex;
use std::sync::Arc;

// Burn imports for tensor operations and NN modules.
use burn::tensor::{Tensor, Distribution};
use burn::tensor::backend::Backend; // Only need the Backend trait.
use burn_ndarray::NdArray; // Our chosen backend for f32

// We import the linear layer from burn::nn.
use burn::nn::{Linear, LinearConfig};

// ================================
// Data Structures for Text Data
// ================================

/// Represents the three phases of wisdom.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Phase {
    Fumigation,
    Awakening,
    Enlightenment,
}

/// Represents a raw affirmation loaded from a text file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Affirmation {
    pub text: String,
}

/// Represents a processed affirmation that has been cleaned, tokenized,
/// and numerically encoded, along with its phase label.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessedAffirmation {
    pub token_ids: Vec<usize>,
    pub tokens: Vec<String>,
    pub phase: Phase,
}

// ================================
// Text Processing Functions
// ================================

fn clean_text(text: &str) -> String {
    let re = Regex::new(r"[^\w\s]").unwrap();
    let no_punct = re.replace_all(text, "");
    let trimmed = no_punct.trim();
    let re_spaces = Regex::new(r"\s+").unwrap();
    re_spaces.replace_all(trimmed, " ").to_string().to_lowercase()
}

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace().map(|s| s.to_string()).collect()
}

fn build_vocab(tokens: &[String]) -> HashMap<String, usize> {
    let mut vocab = HashMap::new();
    let mut id = 1; // Reserve 0 for unknown/padding.
    for token in tokens {
        if !vocab.contains_key(token) {
            vocab.insert(token.clone(), id);
            id += 1;
        }
    }
    vocab
}

fn encode_tokens(tokens: &[String], vocab: &HashMap<String, usize>) -> Vec<usize> {
    tokens.iter().map(|token| *vocab.get(token).unwrap_or(&0)).collect()
}

// ================================
// Custom Mapper for Affirmations
// ================================

struct AffirmationMapper {
    vocab: HashMap<String, usize>,
}

impl AffirmationMapper {
    pub fn new(vocab: HashMap<String, usize>) -> Self {
        Self { vocab }
    }

    pub fn process(&self, affirmation: &Affirmation, phase: Phase) -> ProcessedAffirmation {
        let cleaned = clean_text(&affirmation.text);
        let tokens = tokenize(&cleaned);
        let token_ids = encode_tokens(&tokens, &self.vocab);
        ProcessedAffirmation { token_ids, tokens, phase }
    }
}

// ================================
// Learnable Phi‑Layer
// ================================

/// A learnable phi‑layer that transforms an embedding vector using a non‑linear
/// combination of tanh and sin functions. The transformation is:
///   φ(x) = α * tanh(x) + β * sin(x) + γ
/// where α, β, and γ are learnable scalar parameters (stored as 1‑dimensional tensors).
#[derive(Debug)]
struct PhiLayer<B: Backend> {
    // We store these as 1D tensors of shape [1] so that they broadcast with a vector.
    alpha: Tensor<B, 1>,
    beta: Tensor<B, 1>,
    gamma: Tensor<B, 1>,
}

impl<B: Backend> PhiLayer<B> {
    /// Creates a new phi‑layer with learnable parameters.
    pub fn new(device: &B::Device) -> Self {
        // Create 1D tensors with a single element.
        let alpha = Tensor::<B, 1>::from_data([1.0], device);
        let beta  = Tensor::<B, 1>::from_data([1.0], device);
        let gamma = Tensor::<B, 1>::from_data([0.0], device);
        Self { alpha, beta, gamma }
    }

    /// Forward pass: transforms the input embedding vector.
    pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        let tanh_val = input.clone().tanh();
        let sin_val = input.sin();
        // The multiplication now happens between 1D tensors (which supports broadcasting).
        self.alpha.clone() * tanh_val + self.beta.clone() * sin_val + self.gamma.clone()
    }
}

// ================================
// Embedding Layer with Phi Transformation
// ================================

#[derive(Debug)]
struct EmbeddingLayer<B: Backend> {
    embeddings: Tensor<B, 2>,
    embed_dim: usize,
    phi_layer: PhiLayer<B>,
}

impl<B: Backend> EmbeddingLayer<B> {
    /// Creates a new embedding layer with a learnable phi layer.
    pub fn new(vocab_size: usize, embed_dim: usize, device: &B::Device) -> Self {
        let embeddings = Tensor::random([vocab_size + 1, embed_dim], Distribution::Default, device);
        let phi_layer = PhiLayer::new(device);
        Self { embeddings, embed_dim, phi_layer }
    }

    /// Given token IDs, gathers the corresponding embedding rows, applies the phi transformation,
    /// and stacks the results into a 2D tensor of shape [sequence_length, embed_dim].
    pub fn embed_sequence(&self, token_ids: &[usize], device: &B::Device) -> Tensor<B, 2> {
        let mut gathered_rows = Vec::with_capacity(token_ids.len());
        for &id in token_ids {
            let row = self.gather_row(id, device);
            let transformed = self.phi_layer.forward(row);
            gathered_rows.push(transformed);
        }
        Tensor::stack(gathered_rows, 0)
    }

    /// Helper: gathers a row (i.e. a token’s embedding) from the embedding matrix.
    fn gather_row(&self, index: usize, _device: &B::Device) -> Tensor<B, 1> {
        self.embeddings
            .clone() // Clone to avoid moving out.
            .slice([index..index + 1, 0..self.embed_dim])
            .reshape([self.embed_dim])
    }
}

// ================================
// Model Architecture: Text Classifier
// ================================

#[derive(Debug)]
struct TextClassifier<B: Backend> {
    embedding_layer: EmbeddingLayer<B>,
    classifier: Linear<B>,
}

impl<B: Backend> TextClassifier<B> {
    /// Constructs a new TextClassifier.
    pub fn new(vocab_size: usize, embed_dim: usize, num_classes: usize, device: &B::Device) -> Self {
        let embedding_layer = EmbeddingLayer::<B>::new(vocab_size, embed_dim, device);
        let classifier = LinearConfig::new(embed_dim, num_classes).init(device);
        Self { embedding_layer, classifier }
    }

    /// Forward pass:
    /// - Convert token IDs to embeddings.
    /// - Sum embeddings along the sequence dimension, compute the mean, and squeeze to 1D.
    /// - Pass the resulting vector through the classifier.
    pub fn forward(&self, token_ids: &[usize], device: &B::Device) -> Tensor<B, 1> {
        let embeddings = self.embedding_layer.embed_sequence(token_ids, device);
        // Sum embeddings along dimension 0 (the sequence dimension).
        let sum_embedding = embeddings.clone().sum_dim(0);
        // Compute the number of tokens (as f32).
        let count = embeddings.shape().dims[0] as f32;
        // Divide to get the mean. The result has the same shape as sum_embedding.
        // Squeeze it to obtain a 1D tensor.
        let mean_embedding = (sum_embedding / count).squeeze(0);
        self.classifier.forward(mean_embedding)
    }
}

// ================================
// Data Loading Functions
// ================================

fn load_affirmations(paths: Vec<&str>) -> Vec<(Affirmation, Phase)> {
    let mut affirmations = Vec::new();
    for path in paths {
        let file = File::open(path).expect(&format!("Could not open {}", path));
        let reader = BufReader::new(file);
        let path_lower = path.to_lowercase();
        let phase = if path_lower.contains("fumi") {
            Phase::Fumigation
        } else if path_lower.contains("awak") {
            Phase::Awakening
        } else if path_lower.contains("enlight") {
            Phase::Enlightenment
        } else {
            Phase::Fumigation
        };
        for line in reader.lines().flatten() {
            if !line.trim().is_empty() {
                affirmations.push((Affirmation { text: line }, phase.clone()));
            }
        }
    }
    affirmations
}

// ================================
// Background Training and Interactive Inference
// ================================

/// Simulates background training by periodically printing an update.
/// (Replace this simulation with your real training/optimizer logic.)
async fn background_training_simulation<B: Backend>(
    model: Arc<Mutex<TextClassifier<B>>>,
    _device: B::Device,
) {
    loop {
        sleep(Duration::from_secs(5)).await;
        let _model_lock = model.lock().await;
        println!("Background training: model parameters updated.");
    }
}

/// Handles interactive queries by processing user input,
/// converting it to token IDs, performing a forward pass, and printing logits.
async fn interactive_query_loop<B: Backend>(
    model: Arc<Mutex<TextClassifier<B>>>,
    mapper: &AffirmationMapper,
    device: B::Device,
) {
    println!("Enter a query (press ENTER to exit):");
    loop {
        print!("You: ");
        std::io::stdout().flush().unwrap();
        let mut input_line = String::new();
        std::io::stdin().read_line(&mut input_line).unwrap();
        let query = input_line.trim();
        if query.is_empty() {
            println!("Exiting query loop.");
            break;
        }
        let query_aff = Affirmation { text: query.to_string() };
        // Here we assign a dummy phase (Awakening) to the query.
        let processed_query = mapper.process(&query_aff, Phase::Awakening);
        let model_lock = model.lock().await;
        let logits = model_lock.forward(&processed_query.token_ids, &device);
        println!("Model logits: {:?}", logits);
    }
}

// ================================
// Main Async Function
// ================================

#[tokio::main]
async fn main() {
    // ---------- Data Loading & Processing ----------
    let raw_affirmations = load_affirmations(vec![
        r"C:\RustProjects\tr-ai-training\processed\fumi_corpus.txt",
        r"C:\RustProjects\tr-ai-training\processed\awak_corpus.txt",
        r"C:\RustProjects\tr-ai-training\processed\enlight_corpus.txt",
    ]);

    let mut all_tokens = Vec::new();
    for (aff, _) in &raw_affirmations {
        let cleaned = clean_text(&aff.text);
        let tokens = tokenize(&cleaned);
        all_tokens.extend(tokens);
    }
    let vocab = build_vocab(&all_tokens);
    let mapper = AffirmationMapper::new(vocab);
    let processed: Vec<ProcessedAffirmation> = raw_affirmations
        .iter()
        .map(|(aff, phase)| mapper.process(aff, phase.clone()))
        .collect();

    // (Optional) Save processed data as JSON grouped by phase.
    let mut grouped: HashMap<Phase, Vec<ProcessedAffirmation>> = HashMap::new();
    for pa in &processed {
        grouped.entry(pa.phase.clone()).or_default().push(pa.clone());
    }
    let out_dir = "processed_output";
    std::fs::create_dir_all(out_dir).expect("Failed to create output directory");
    for (phase, affirmations) in &grouped {
        let filename = match phase {
            Phase::Fumigation => "processed_fumi.json",
            Phase::Awakening => "processed_awak.json",
            Phase::Enlightenment => "processed_enlight.json",
        };
        let out_path = format!("{}/{}", out_dir, filename);
        let file = File::create(&out_path).expect("Failed to create output file");
        serde_json::to_writer_pretty(file, &affirmations)
            .expect("Failed to write JSON to file");
        println!("Saved {} affirmations to {}", filename, out_path);
    }

    // ---------- Model Initialization ----------
    let embed_dim = 128;
    // Compute vocabulary size as the maximum token ID (alternatively, you could use vocab.len()).
    let vocab_size = all_tokens.iter().fold(0, |max, token| {
        let id = *build_vocab(&all_tokens).get(token).unwrap_or(&0);
        if id > max { id } else { max }
    });
    let device = <NdArray<f32> as Backend>::Device::default();
    let num_classes = 4; // For example, 4 classes for classification.
    let classifier = TextClassifier::<NdArray<f32>>::new(vocab_size, embed_dim, num_classes, &device);

    // Wrap the classifier in an Arc<Mutex<...>> so it can be shared between tasks.
    let shared_model = Arc::new(Mutex::new(classifier));

    // ---------- Spawn Background Training Task ----------
    let bg_model = Arc::clone(&shared_model);
    let bg_device = device.clone();
    tokio::spawn(async move {
        background_training_simulation(bg_model, bg_device).await;
    });

    // ---------- Interactive Query Loop ----------
    interactive_query_loop(shared_model, &mapper, device).await;

    
}


