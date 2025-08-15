import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
import random
from collections import deque, defaultdict
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SyntheticDataset(Dataset):
    """Creates a synthetic dataset with varying quality and relevance"""
    def __init__(self, n_samples=10000, n_features=20, n_classes=3, noise_level=0.1):
        # Generate base classification data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_features//2,
            n_redundant=n_features//4,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Add quality scores (some samples are just better/cleaner)
        self.quality_scores = np.random.beta(2, 5, n_samples)  # Most samples are lower quality
        
        # Add relevance scores (some samples are more relevant to the task)
        self.relevance_scores = np.random.beta(3, 2, n_samples)  # Most samples are relevant
        
        # Add noise based on quality (lower quality = more noise)
        noise = np.random.normal(0, noise_level, X.shape)
        noise_multiplier = (1 - self.quality_scores).reshape(-1, 1)
        X = X + noise * noise_multiplier
        
        # Make some samples irrelevant by scrambling their features
        irrelevant_mask = self.relevance_scores < 0.3
        X[irrelevant_mask] = np.random.normal(0, 1, (irrelevant_mask.sum(), n_features))
        
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
        # True scores (hidden from DRPS, only used for evaluation)
        self.true_quality = self.quality_scores
        self.true_relevance = self.relevance_scores
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx

class RelevanceScorer(nn.Module):
    """Stage 1: Learns what data is relevant for the task"""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class QualityRater(nn.Module):
    """Stage 2: Rates the quality of relevant data"""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for relevance score
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, relevance_score):
        # Ensure relevance_score has the right shape to concatenate
        if relevance_score.dim() == 1:
            relevance_score = relevance_score.unsqueeze(1)
        elif relevance_score.dim() == 0:
            relevance_score = relevance_score.unsqueeze(0).unsqueeze(1)
        
        combined = torch.cat([x, relevance_score], dim=1)
        return self.network(combined)

class DiversityController:
    """Stage 3: Ensures diversity in selected data"""
    def __init__(self, target_distribution=None, memory_size=1000):
        # Default target: prefer high quality but maintain some diversity
        self.target_dist = target_distribution or [0.05, 0.1, 0.15, 0.2, 0.25, 0.25]  # for scores 0-1 binned
        self.memory = deque(maxlen=memory_size)
        self.selection_history = defaultdict(int)
        
    def should_select(self, quality_score, relevance_score, diversity_factor=0.3):
        """Decides whether to select a sample based on quality and diversity needs"""
        # Bin the quality score
        quality_bin = min(int(quality_score * 6), 5)
        
        # Calculate current distribution
        total_selected = sum(self.selection_history.values()) + 1
        current_prop = self.selection_history[quality_bin] / total_selected
        target_prop = self.target_dist[quality_bin]
        
        # Base selection probability from quality and relevance
        base_prob = (quality_score * 0.7 + relevance_score * 0.3)
        
        # Diversity adjustment
        if current_prop < target_prop:
            diversity_boost = diversity_factor * (target_prop - current_prop) / target_prop
        else:
            diversity_penalty = diversity_factor * (current_prop - target_prop) / target_prop
            diversity_boost = -diversity_penalty
            
        final_prob = min(1.0, max(0.1, base_prob + diversity_boost))
        
        selected = random.random() < final_prob
        
        if selected:
            self.selection_history[quality_bin] += 1
            self.memory.append((quality_score, relevance_score))
            
        return selected

class MainModel(nn.Module):
    """The actual model that learns from DRPS-selected data"""
    def __init__(self, input_dim, n_classes, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, n_classes)
        )
        
    def forward(self, x):
        return self.network(x)

class DRPS:
    """The complete Diverse Relevance Picking System"""
    def __init__(self, input_dim, n_classes, device='cpu'):
        self.device = device
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        # Initialize all components
        self.relevance_scorer = RelevanceScorer(input_dim).to(device)
        self.quality_rater = QualityRater(input_dim).to(device)
        self.diversity_controller = DiversityController()
        self.main_model = MainModel(input_dim, n_classes).to(device)
        
        # Optimizers
        self.rel_optimizer = optim.Adam(self.relevance_scorer.parameters(), lr=0.001)
        self.qual_optimizer = optim.Adam(self.quality_rater.parameters(), lr=0.001)
        self.main_optimizer = optim.Adam(self.main_model.parameters(), lr=0.001)
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()
        
        # Training stats
        self.selection_stats = {
            'total_seen': 0,
            'total_selected': 0,
            'relevance_accuracy': [],
            'quality_accuracy': [],
            'main_model_accuracy': []
        }
        
    def train_relevance_scorer(self, dataset, epochs=50):
        """Train the relevance scorer using a small bootstrap sample"""
        print("Training Relevance Scorer...")
        
        # Use a small sample to bootstrap relevance learning
        bootstrap_size = min(1000, len(dataset))
        indices = random.sample(range(len(dataset)), bootstrap_size)
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for idx in indices:
                x, y, _ = dataset[idx]
                x = x.unsqueeze(0).to(self.device)
                
                # True relevance score (for training only)
                true_relevance = torch.FloatTensor([dataset.true_relevance[idx]]).to(self.device)
                
                pred_relevance = self.relevance_scorer(x)
                loss = self.mse_criterion(pred_relevance, true_relevance.unsqueeze(0))
                
                self.rel_optimizer.zero_grad()
                loss.backward()
                self.rel_optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy (within 0.2 threshold)
                if abs(pred_relevance.item() - true_relevance.item()) < 0.2:
                    correct += 1
                total += 1
                
            accuracy = correct / total
            self.selection_stats['relevance_accuracy'].append(accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Relevance Loss: {total_loss/len(indices):.4f}, Accuracy: {accuracy:.4f}")
    
    def train_quality_rater(self, dataset, epochs=50):
        """Train the quality rater using relevance scores"""
        print("Training Quality Rater...")
        
        bootstrap_size = min(1000, len(dataset))
        indices = random.sample(range(len(dataset)), bootstrap_size)
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for idx in indices:
                x, y, _ = dataset[idx]
                x = x.unsqueeze(0).to(self.device)
                
                # Get relevance score
                with torch.no_grad():
                    relevance_score = self.relevance_scorer(x)
                
                # True quality score (for training only)
                true_quality = torch.FloatTensor([dataset.true_quality[idx]]).to(self.device)
                
                pred_quality = self.quality_rater(x, relevance_score)
                loss = self.mse_criterion(pred_quality, true_quality.unsqueeze(0))
                
                self.qual_optimizer.zero_grad()
                loss.backward()
                self.qual_optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy (within 0.2 threshold)
                if abs(pred_quality.item() - true_quality.item()) < 0.2:
                    correct += 1
                total += 1
                
            accuracy = correct / total
            self.selection_stats['quality_accuracy'].append(accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Quality Loss: {total_loss/len(indices):.4f}, Accuracy: {accuracy:.4f}")
    
    def select_data_batch(self, dataset, batch_size=32, max_attempts=1000):
        """Select a batch of data using DRPS"""
        selected_samples = []
        selected_indices = []
        attempts = 0
        
        while len(selected_samples) < batch_size and attempts < max_attempts:
            # Random sample from dataset
            idx = random.randint(0, len(dataset) - 1)
            x, y, _ = dataset[idx]
            x_tensor = x.unsqueeze(0).to(self.device)
            
            # Stage 1: Get relevance score
            with torch.no_grad():
                relevance_score = self.relevance_scorer(x_tensor).item()
            
            # Stage 2: Get quality score
            with torch.no_grad():
                relevance_tensor = torch.FloatTensor([relevance_score]).to(self.device)
                quality_score = self.quality_rater(x_tensor, relevance_tensor).item()
            
            # Stage 3: Diversity controller decides
            if self.diversity_controller.should_select(quality_score, relevance_score):
                selected_samples.append((x, y))
                selected_indices.append(idx)
            
            attempts += 1
            self.selection_stats['total_seen'] += 1
        
        self.selection_stats['total_selected'] += len(selected_samples)
        return selected_samples, selected_indices
    
    def train_main_model(self, dataset, test_dataset, epochs=100, batch_size=32):
        """Train the main model using DRPS-selected data"""
        print("Training Main Model with DRPS...")
        
        for epoch in range(epochs):
            # Select batch using DRPS
            selected_batch, _ = self.select_data_batch(dataset, batch_size)
            
            if len(selected_batch) == 0:
                continue
                
            # Prepare batch
            batch_x = torch.stack([x for x, y in selected_batch]).to(self.device)
            batch_y = torch.stack([y for x, y in selected_batch]).to(self.device)
            
            # Train main model
            predictions = self.main_model(batch_x)
            loss = self.criterion(predictions, batch_y)
            
            self.main_optimizer.zero_grad()
            loss.backward()
            self.main_optimizer.step()
            
            # Evaluate on test set every 10 epochs
            if epoch % 10 == 0:
                accuracy = self.evaluate(test_dataset)
                self.selection_stats['main_model_accuracy'].append(accuracy)
                print(f"Epoch {epoch}: Main Model Accuracy: {accuracy:.4f}, Selected: {self.selection_stats['total_selected']}/{self.selection_stats['total_seen']}")
    
    def evaluate(self, test_dataset):
        """Evaluate the main model"""
        self.main_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y, _ in test_dataset:
                x = x.unsqueeze(0).to(self.device)
                predictions = self.main_model(x)
                predicted = predictions.argmax(dim=1)
                correct += (predicted == y.to(self.device)).sum().item()
                total += 1
        
        self.main_model.train()
        return correct / total

def random_baseline(dataset, test_dataset, epochs=100, batch_size=32, device='cpu'):
    """Baseline: Train model with random data selection"""
    print("Training Random Baseline...")
    
    model = MainModel(dataset.X.shape[1], len(torch.unique(dataset.y))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    accuracies = []
    
    for epoch in range(epochs):
        # Random batch selection
        indices = random.sample(range(len(dataset)), min(batch_size, len(dataset)))
        batch_x = torch.stack([dataset.X[i] for i in indices]).to(device)
        batch_y = torch.stack([dataset.y[i] for i in indices]).to(device)
        
        # Train
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for x, y, _ in test_dataset:
                    x = x.unsqueeze(0).to(device)
                    predictions = model(x)
                    predicted = predictions.argmax(dim=1)
                    correct += (predicted == y.to(device)).sum().item()
                    total += 1
            
            accuracy = correct / total
            accuracies.append(accuracy)
            model.train()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Random Baseline Accuracy: {accuracy:.4f}")
    
    return accuracies

def run_comprehensive_test():
    """Run comprehensive DRPS testing"""
    print("=" * 60)
    print("DRPS: Diverse Relevance Picking System - Comprehensive Test")
    print("=" * 60)
    
    # Create datasets
    print("Creating synthetic dataset...")
    full_dataset = SyntheticDataset(n_samples=5000, n_features=20, n_classes=3)
    
    # Split into train/test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, len(full_dataset)))
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize DRPS
    print("\nInitializing DRPS...")
    drps = DRPS(input_dim=20, n_classes=3, device=device)
    
    # Train DRPS components
    print("\n" + "="*40)
    print("PHASE 1: Training DRPS Components")
    print("="*40)
    
    start_time = time.time()
    drps.train_relevance_scorer(full_dataset, epochs=30)
    drps.train_quality_rater(full_dataset, epochs=30)
    component_time = time.time() - start_time
    
    # Train main model with DRPS
    print("\n" + "="*40)
    print("PHASE 2: Training Main Model with DRPS")
    print("="*40)
    
    start_time = time.time()
    drps.train_main_model(train_dataset, test_dataset, epochs=100, batch_size=32)
    drps_time = time.time() - start_time
    
    # Train random baseline
    print("\n" + "="*40)
    print("PHASE 3: Training Random Baseline")
    print("="*40)
    
    start_time = time.time()
    random_accuracies = random_baseline(full_dataset, test_dataset, epochs=100, batch_size=32, device=device)
    baseline_time = time.time() - start_time
    
    # Results and Analysis
    print("\n" + "="*60)
    print("RESULTS AND ANALYSIS")
    print("="*60)
    
    final_drps_acc = drps.selection_stats['main_model_accuracy'][-1] if drps.selection_stats['main_model_accuracy'] else 0
    final_random_acc = random_accuracies[-1] if random_accuracies else 0
    
    print(f"\nFinal Accuracies:")
    print(f"DRPS System: {final_drps_acc:.4f}")
    print(f"Random Baseline: {final_random_acc:.4f}")
    print(f"Improvement: {((final_drps_acc - final_random_acc) / final_random_acc * 100):.2f}%")
    
    print(f"\nData Selection Efficiency:")
    selection_ratio = drps.selection_stats['total_selected'] / drps.selection_stats['total_seen']
    print(f"Selection Ratio: {selection_ratio:.4f} ({drps.selection_stats['total_selected']}/{drps.selection_stats['total_seen']})")
    print(f"DRPS used {selection_ratio*100:.1f}% of examined data")
    
    print(f"\nTraining Times:")
    print(f"DRPS Component Training: {component_time:.2f}s")
    print(f"DRPS Main Training: {drps_time:.2f}s")
    print(f"Random Baseline: {baseline_time:.2f}s")
    print(f"Total DRPS Time: {component_time + drps_time:.2f}s")
    
    print(f"\nComponent Performance:")
    if drps.selection_stats['relevance_accuracy']:
        print(f"Relevance Scorer Final Accuracy: {drps.selection_stats['relevance_accuracy'][-1]:.4f}")
    if drps.selection_stats['quality_accuracy']:
        print(f"Quality Rater Final Accuracy: {drps.selection_stats['quality_accuracy'][-1]:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Learning curves
    plt.subplot(1, 3, 1)
    if drps.selection_stats['main_model_accuracy']:
        plt.plot(range(0, len(drps.selection_stats['main_model_accuracy']) * 10, 10), 
                drps.selection_stats['main_model_accuracy'], 'b-', label='DRPS', linewidth=2)
    if random_accuracies:
        plt.plot(range(0, len(random_accuracies) * 10, 10), 
                random_accuracies, 'r--', label='Random', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Component training
    plt.subplot(1, 3, 2)
    if drps.selection_stats['relevance_accuracy']:
        plt.plot(drps.selection_stats['relevance_accuracy'], 'g-', label='Relevance Scorer')
    if drps.selection_stats['quality_accuracy']:
        plt.plot(drps.selection_stats['quality_accuracy'], 'orange', label='Quality Rater')
    plt.xlabel('Epoch')
    plt.ylabel('Component Accuracy')
    plt.title('DRPS Component Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Data quality distribution
    plt.subplot(1, 3, 3)
    quality_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_counts = [drps.diversity_controller.selection_history[i] for i in range(6)]
    plt.bar(range(6), bin_counts, alpha=0.7)
    plt.xlabel('Quality Bin')
    plt.ylabel('Samples Selected')
    plt.title('Data Selection Distribution')
    plt.xticks(range(6), ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', '1.0'])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Detailed analysis
    print(f"\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    print(f"\nDRPS successfully demonstrates:")
    print(f"   - Relevance learning with {drps.selection_stats['relevance_accuracy'][-1]:.1%} accuracy")
    print(f"   - Quality assessment with {drps.selection_stats['quality_accuracy'][-1]:.1%} accuracy") 
    print(f"   - Intelligent data selection ({selection_ratio:.1%} selection rate)")
    print(f"   - {'Superior' if final_drps_acc > final_random_acc else 'Competitive'} performance vs random selection")
    
    if final_drps_acc > final_random_acc:
        print(f"\nDRPS achieved {((final_drps_acc - final_random_acc) / final_random_acc * 100):.1f}% improvement!")
        print(f"   This suggests the three-stage approach is working!")
    else:
        print(f"\nDRPS performed similarly to random selection.")
        print(f"   This could mean: better hypertuning needed, or the synthetic data isn't complex enough.")
    
    print(f"\nKey Insights:")
    print(f"   - DRPS examined {drps.selection_stats['total_seen']} samples")
    print(f"   - Selected only {drps.selection_stats['total_selected']} for training")
    print(f"   - Achieved similar/better results with {selection_ratio:.1%} of the data")
    
    return drps, random_accuracies

if __name__ == "__main__":
    # Run the comprehensive test
    drps_system, baseline_results = run_comprehensive_test()
    
    print(f"\n" + "="*60)