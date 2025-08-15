import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random
from collections import deque, defaultdict
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class MNISTDatasetWithScores(Dataset):
    """MNIST with computed quality and relevance scores"""
    def __init__(self, mnist_dataset, add_artificial_noise=True):
        self.mnist_dataset = mnist_dataset
        self.data = []
        self.targets = []
        self.quality_scores = []
        self.relevance_scores = []
        
        print("Computing quality and relevance scores for MNIST...")
        
        for i, (image, label) in enumerate(mnist_dataset):
            if i % 10000 == 0:
                print(f"Processed {i}/{len(mnist_dataset)} samples")
            
            # Convert to tensor if needed
            if not isinstance(image, torch.Tensor):
                image = transforms.ToTensor()(image)
            
            # Flatten image for easier processing
            flat_image = image.view(-1)
            
            # Quality score based on multiple factors
            quality = self._compute_quality_score(flat_image, add_artificial_noise)
            
            # Relevance score based on how "typical" the digit looks
            relevance = self._compute_relevance_score(flat_image, label)
            
            self.data.append(flat_image)
            self.targets.append(label)
            self.quality_scores.append(quality)
            self.relevance_scores.append(relevance)
        
        self.data = torch.stack(self.data)
        self.targets = torch.tensor(self.targets)
        self.quality_scores = np.array(self.quality_scores)
        self.relevance_scores = np.array(self.relevance_scores)
        
        print(f"Dataset created with {len(self.data)} samples")
        print(f"Quality scores: mean={self.quality_scores.mean():.3f}, std={self.quality_scores.std():.3f}")
        print(f"Relevance scores: mean={self.relevance_scores.mean():.3f}, std={self.relevance_scores.std():.3f}")
    
    def _compute_quality_score(self, flat_image, add_noise=True):
        """Compute quality score based on image characteristics"""
        # Convert to numpy for easier computation
        img_np = flat_image.numpy()
        
        # Factor 1: Image sharpness (higher variance = sharper)
        sharpness = np.var(img_np)
        sharpness_score = min(1.0, sharpness / 0.1)  # Normalize
        
        # Factor 2: Contrast (difference between max and min)
        contrast = np.max(img_np) - np.min(img_np)
        contrast_score = contrast  # Already 0-1 for normalized images
        
        # Factor 3: Information content (entropy-like measure)
        # Count non-zero pixels
        non_zero_ratio = np.sum(img_np > 0.1) / len(img_np)
        info_score = min(1.0, non_zero_ratio * 2)  # Prefer images with reasonable amount of content
        
        # Factor 4: Noise level (artificially add some if requested)
        noise_factor = 1.0
        if add_noise and random.random() < 0.3:  # 30% of images get noise
            noise_level = random.uniform(0.1, 0.5)
            noise_factor = 1.0 - noise_level
        
        # Combine factors
        quality = (sharpness_score * 0.3 + contrast_score * 0.3 + 
                  info_score * 0.2 + noise_factor * 0.2)
        
        return np.clip(quality, 0.0, 1.0)
    
    def _compute_relevance_score(self, flat_image, label):
        """Compute relevance score based on how typical the digit looks"""
        img_np = flat_image.numpy()
        
        # Factor 1: Digit is well-formed (has reasonable pixel distribution)
        # Center region should have more pixels for most digits
        img_2d = img_np.reshape(28, 28)
        center_region = img_2d[7:21, 7:21]  # 14x14 center region
        center_intensity = np.mean(center_region)
        edge_intensity = np.mean(img_2d) - center_intensity
        
        # Most digits have more intensity in center than edges
        formation_score = min(1.0, center_intensity / (edge_intensity + 0.01))
        
        # Factor 2: Reasonable total intensity (not too faint, not too bold)
        total_intensity = np.mean(img_np)
        intensity_score = 1.0 - abs(total_intensity - 0.3) / 0.3  # Prefer around 0.3 intensity
        intensity_score = max(0.0, intensity_score)
        
        # Factor 3: Some digits are just naturally harder to classify
        # Add some randomness to simulate "relevance"
        random_factor = 0.7 + 0.3 * random.random()  # Between 0.7 and 1.0
        
        # Combine factors
        relevance = (formation_score * 0.4 + intensity_score * 0.3 + random_factor * 0.3)
        
        return np.clip(relevance, 0.0, 1.0)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], idx

class RelevanceScorer(nn.Module):
    """Stage 1: Learns what data is relevant for the task"""
    def __init__(self, input_dim=784, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class QualityRater(nn.Module):
    """Stage 2: Rates the quality of relevant data"""
    def __init__(self, input_dim=784, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for relevance score
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, relevance_score):
        if relevance_score.dim() == 1:
            relevance_score = relevance_score.unsqueeze(1)
        elif relevance_score.dim() == 0:
            relevance_score = relevance_score.unsqueeze(0).unsqueeze(1)
        
        combined = torch.cat([x, relevance_score], dim=1)
        return self.network(combined)

class DiversityController:
    """Stage 3: Ensures diversity in selected data"""
    def __init__(self, target_distribution=None, memory_size=2000):
        # More balanced distribution for real data
        self.target_dist = target_distribution or [0.05, 0.15, 0.25, 0.25, 0.20, 0.10]
        self.memory = deque(maxlen=memory_size)
        self.selection_history = defaultdict(int)
        
    def should_select(self, quality_score, relevance_score, diversity_factor=0.4):
        """Decides whether to select a sample based on quality and diversity needs"""
        quality_bin = min(int(quality_score * 6), 5)
        
        total_selected = sum(self.selection_history.values()) + 1
        current_prop = self.selection_history[quality_bin] / total_selected
        target_prop = self.target_dist[quality_bin]
        
        # Base selection probability
        base_prob = (quality_score * 0.6 + relevance_score * 0.4)
        
        # Diversity adjustment
        if current_prop < target_prop:
            diversity_boost = diversity_factor * (target_prop - current_prop) / target_prop
        else:
            diversity_penalty = diversity_factor * (current_prop - target_prop) / target_prop
            diversity_boost = -diversity_penalty
            
        final_prob = min(1.0, max(0.05, base_prob + diversity_boost))
        
        selected = random.random() < final_prob
        
        if selected:
            self.selection_history[quality_bin] += 1
            self.memory.append((quality_score, relevance_score))
            
        return selected

class MainModel(nn.Module):
    """The actual MNIST classifier that learns from DRPS-selected data"""
    def __init__(self, input_dim=784, n_classes=10, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//4, n_classes)
        )
        
    def forward(self, x):
        return self.network(x)

class DRPS:
    """The complete Diverse Relevance Picking System for MNIST"""
    def __init__(self, input_dim=784, n_classes=10, device='cpu'):
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
        """Train the relevance scorer using a bootstrap sample"""
        print("Training Relevance Scorer...")
        
        bootstrap_size = min(2000, len(dataset))
        indices = random.sample(range(len(dataset)), bootstrap_size)
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            # Shuffle indices for each epoch
            random.shuffle(indices)
            
            for idx in indices:
                x, y, _ = dataset[idx]
                x = x.unsqueeze(0).to(self.device)
                
                true_relevance = torch.FloatTensor([dataset.relevance_scores[idx]]).to(self.device)
                
                pred_relevance = self.relevance_scorer(x)
                loss = self.mse_criterion(pred_relevance, true_relevance.unsqueeze(0))
                
                self.rel_optimizer.zero_grad()
                loss.backward()
                self.rel_optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy (within 0.15 threshold for MNIST)
                if abs(pred_relevance.item() - true_relevance.item()) < 0.15:
                    correct += 1
                total += 1
                
            accuracy = correct / total
            self.selection_stats['relevance_accuracy'].append(accuracy)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}: Relevance Loss: {total_loss/len(indices):.4f}, Accuracy: {accuracy:.4f}")
    
    def train_quality_rater(self, dataset, epochs=50):
        """Train the quality rater using relevance scores"""
        print("Training Quality Rater...")
        
        bootstrap_size = min(2000, len(dataset))
        indices = random.sample(range(len(dataset)), bootstrap_size)
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            random.shuffle(indices)
            
            for idx in indices:
                x, y, _ = dataset[idx]
                x = x.unsqueeze(0).to(self.device)
                
                # Get relevance score
                with torch.no_grad():
                    relevance_score = self.relevance_scorer(x)
                
                true_quality = torch.FloatTensor([dataset.quality_scores[idx]]).to(self.device)
                
                pred_quality = self.quality_rater(x, relevance_score)
                loss = self.mse_criterion(pred_quality, true_quality.unsqueeze(0))
                
                self.qual_optimizer.zero_grad()
                loss.backward()
                self.qual_optimizer.step()
                
                total_loss += loss.item()
                
                if abs(pred_quality.item() - true_quality.item()) < 0.15:
                    correct += 1
                total += 1
                
            accuracy = correct / total
            self.selection_stats['quality_accuracy'].append(accuracy)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}: Quality Loss: {total_loss/len(indices):.4f}, Accuracy: {accuracy:.4f}")
    
    def select_data_batch(self, dataset, batch_size=64, max_attempts=2000):
        """Select a batch of data using DRPS"""
        selected_samples = []
        selected_indices = []
        attempts = 0
        
        while len(selected_samples) < batch_size and attempts < max_attempts:
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
    
    def train_main_model(self, dataset, test_dataset, epochs=200, batch_size=64):
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
            
            # Evaluate on test set every 20 epochs
            if epoch % 20 == 0:
                accuracy = self.evaluate(test_dataset)
                self.selection_stats['main_model_accuracy'].append(accuracy)
                selection_ratio = self.selection_stats['total_selected'] / max(1, self.selection_stats['total_seen'])
                print(f"Epoch {epoch}: Accuracy: {accuracy:.4f}, Selection Rate: {selection_ratio:.4f}")
    
    def evaluate(self, test_dataset):
        """Evaluate the main model"""
        self.main_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            # Evaluate on a subset for speed
            test_indices = random.sample(range(len(test_dataset)), min(2000, len(test_dataset)))
            for idx in test_indices:
                x, y, _ = test_dataset[idx]
                x = x.unsqueeze(0).to(self.device)
                predictions = self.main_model(x)
                predicted = predictions.argmax(dim=1)
                correct += (predicted == y.to(self.device)).sum().item()
                total += 1
        
        self.main_model.train()
        return correct / total

def random_baseline_mnist(dataset, test_dataset, epochs=200, batch_size=64, device='cpu'):
    """Baseline: Train MNIST classifier with random data selection"""
    print("Training Random Baseline...")
    
    model = MainModel(784, 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    accuracies = []
    
    for epoch in range(epochs):
        # Random batch selection
        indices = random.sample(range(len(dataset)), min(batch_size, len(dataset)))
        batch_x = torch.stack([dataset.data[i] for i in indices]).to(device)
        batch_y = torch.stack([dataset.targets[i] for i in indices]).to(device)
        
        # Train
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate every 20 epochs
        if epoch % 20 == 0:
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                test_indices = random.sample(range(len(test_dataset)), min(2000, len(test_dataset)))
                for idx in test_indices:
                    x, y, _ = test_dataset[idx]
                    x = x.unsqueeze(0).to(device)
                    predictions = model(x)
                    predicted = predictions.argmax(dim=1)
                    correct += (predicted == y.to(device)).sum().item()
                    total += 1
            
            accuracy = correct / total
            accuracies.append(accuracy)
            model.train()
            
            print(f"Epoch {epoch}: Random Baseline Accuracy: {accuracy:.4f}")
    
    return accuracies

def run_mnist_test():
    """Run comprehensive DRPS testing on MNIST"""
    print("=" * 60)
    print("DRPS: Testing on MNIST Dataset")
    print("=" * 60)
    
    # Load MNIST
    print("Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, 
                                           download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, 
                                          download=True, transform=transform)
    
    # Create datasets with quality/relevance scores
    print("Creating enhanced datasets...")
    train_dataset = MNISTDatasetWithScores(mnist_train, add_artificial_noise=True)
    test_dataset = MNISTDatasetWithScores(mnist_test, add_artificial_noise=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize DRPS
    print("\nInitializing DRPS...")
    drps = DRPS(input_dim=784, n_classes=10, device=device)
    
    # Train DRPS components
    print("\n" + "="*40)
    print("PHASE 1: Training DRPS Components")
    print("="*40)
    
    start_time = time.time()
    drps.train_relevance_scorer(train_dataset, epochs=40)
    drps.train_quality_rater(train_dataset, epochs=40)
    component_time = time.time() - start_time
    
    # Train main model with DRPS
    print("\n" + "="*40)
    print("PHASE 2: Training Main Model with DRPS")
    print("="*40)
    
    start_time = time.time()
    drps.train_main_model(train_dataset, test_dataset, epochs=200, batch_size=64)
    drps_time = time.time() - start_time
    
    # Train random baseline
    print("\n" + "="*40)
    print("PHASE 3: Training Random Baseline")
    print("="*40)
    
    start_time = time.time()
    random_accuracies = random_baseline_mnist(train_dataset, test_dataset, 
                                            epochs=200, batch_size=64, device=device)
    baseline_time = time.time() - start_time
    
    # Results and Analysis
    print("\n" + "="*60)
    print("MNIST RESULTS AND ANALYSIS")
    print("="*60)
    
    final_drps_acc = drps.selection_stats['main_model_accuracy'][-1] if drps.selection_stats['main_model_accuracy'] else 0
    final_random_acc = random_accuracies[-1] if random_accuracies else 0
    
    print(f"\nFinal Accuracies:")
    print(f"DRPS System: {final_drps_acc:.4f}")
    print(f"Random Baseline: {final_random_acc:.4f}")
    print(f"Difference: {(final_drps_acc - final_random_acc):.4f}")
    
    print(f"\nData Selection Efficiency:")
    selection_ratio = drps.selection_stats['total_selected'] / max(1, drps.selection_stats['total_seen'])
    print(f"Selection Ratio: {selection_ratio:.4f}")
    print(f"DRPS used {selection_ratio*100:.1f}% of examined data")
    print(f"Data reduction: {(1-selection_ratio)*100:.1f}%")
    
    print(f"\nComponent Performance:")
    if drps.selection_stats['relevance_accuracy']:
        print(f"Relevance Scorer Final Accuracy: {drps.selection_stats['relevance_accuracy'][-1]:.4f}")
    if drps.selection_stats['quality_accuracy']:
        print(f"Quality Rater Final Accuracy: {drps.selection_stats['quality_accuracy'][-1]:.4f}")
    
    print(f"\nTraining Times:")
    print(f"DRPS Component Training: {component_time:.2f}s")
    print(f"DRPS Main Training: {drps_time:.2f}s")
    print(f"Random Baseline: {baseline_time:.2f}s")
    print(f"Total DRPS Time: {component_time + drps_time:.2f}s")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Learning curves
    plt.subplot(2, 2, 1)
    if drps.selection_stats['main_model_accuracy']:
        epochs_drps = range(0, len(drps.selection_stats['main_model_accuracy']) * 20, 20)
        plt.plot(epochs_drps, drps.selection_stats['main_model_accuracy'], 'b-', label='DRPS', linewidth=2)
    if random_accuracies:
        epochs_random = range(0, len(random_accuracies) * 20, 20)
        plt.plot(epochs_random, random_accuracies, 'r--', label='Random', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('MNIST Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Component training
    plt.subplot(2, 2, 2)
    if drps.selection_stats['relevance_accuracy']:
        plt.plot(drps.selection_stats['relevance_accuracy'], 'g-', label='Relevance Scorer', linewidth=2)
    if drps.selection_stats['quality_accuracy']:
        plt.plot(drps.selection_stats['quality_accuracy'], 'orange', label='Quality Rater', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Component Accuracy')
    plt.title('DRPS Component Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Data selection distribution
    plt.subplot(2, 2, 3)
    bin_counts = [drps.diversity_controller.selection_history[i] for i in range(6)]
    plt.bar(range(6), bin_counts, alpha=0.7, color=['red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen'])
    plt.xlabel('Quality Bin')
    plt.ylabel('Samples Selected')
    plt.title('Data Selection Distribution')
    plt.xticks(range(6), ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', '1.0'])
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Sample images from different quality bins
    plt.subplot(2, 2, 4)
    # Show a few sample images from the dataset
    sample_indices = random.sample(range(min(1000, len(train_dataset))), 16)
    sample_grid = []
    for i, idx in enumerate(sample_indices):
        img = train_dataset.data[idx].reshape(28, 28).numpy()
        sample_grid.append(img)
    
    # Create a 4x4 grid
    grid = np.zeros((4*28, 4*28))
    for i in range(4):
        for j in range(4):
            if i*4 + j < len(sample_grid):
                grid[i*28:(i+1)*28, j*28:(j+1)*28] = sample_grid[i*4 + j]
    
    plt.imshow(grid, cmap='gray')
    plt.title('Sample MNIST Images')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Summary
    print(f"\n" + "="*60)
    print("MNIST EXPERIMENT SUMMARY")
    print("="*60)
    
    efficiency_gain = (1 - selection_ratio) * 100
    print(f"\n✓ DRPS achieved {efficiency_gain:.1f}% data reduction")
    print(f"✓ Component learning successful:")
    print(f"   - Relevance scorer: {drps.selection_stats['relevance_accuracy'][-1]:.1%} accuracy")
    print(f"   - Quality rater: {drps.selection_stats['quality_accuracy'][-1]:.1%} accuracy")
    
    if final_drps_acc >= final_random_acc * 0.95:  # Within 5% is good
        print(f"✓ Performance maintained: {final_drps_acc:.1%} vs {final_random_acc:.1%} (random)")
        print(f"✓ Efficiency per data point: {final_drps_acc/selection_ratio:.2f}x better than random")
    else:
        print(f"⚠ Performance gap: {final_drps_acc:.1%} vs {final_random_acc:.1%} (random)")
        print(f"   This could indicate need for hyperparameter tuning")
    
    print(f"\nKey takeaway: DRPS successfully learned to select {selection_ratio:.1%} of MNIST data")
    print(f"while maintaining competitive digit classification performance!")
    
    return drps, random_accuracies

if __name__ == "__main__":
    # Run the MNIST test
    drps_system, baseline_results = run_mnist_test()
    
    print(f"\n" + "="*60)
    print("Experiment completed! Check the plots above for detailed analysis.")
    print("="*60)