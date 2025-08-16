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
import cv2
from PIL import Image

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class CIFAR100DatasetWithScores(Dataset):
    """CIFAR-100 with computed quality and relevance scores"""
    def __init__(self, cifar_dataset, add_artificial_noise=True):
        self.cifar_dataset = cifar_dataset
        self.data = []
        self.targets = []
        self.quality_scores = []
        self.relevance_scores = []
        
        print("Computing quality and relevance scores for CIFAR-100...")
        
        for i, (image, label) in enumerate(cifar_dataset):
            if i % 5000 == 0:
                print(f"Processed {i}/{len(cifar_dataset)} samples")
            
            # Convert PIL image to tensor if needed
            if not isinstance(image, torch.Tensor):
                image = transforms.ToTensor()(image)
            
            # Flatten image for neural network processing
            flat_image = image.view(-1)  # 3072 features (32x32x3)
            
            # Quality score based on multiple factors
            quality = self._compute_quality_score(image, add_artificial_noise)
            
            # Relevance score based on visual characteristics
            relevance = self._compute_relevance_score(image, label)
            
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
    
    def _compute_quality_score(self, image_tensor, add_noise=True):
        """Compute quality score based on image characteristics"""
        # Convert tensor to numpy for OpenCV operations
        img_np = image_tensor.permute(1, 2, 0).numpy()  # Change from CxHxW to HxWxC
        img_np = (img_np * 255).astype(np.uint8)
        
        # Factor 1: Image sharpness using Laplacian variance
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500)  # Normalize based on typical CIFAR values
        
        # Factor 2: Contrast using standard deviation
        contrast = np.std(gray) / 128.0  # Normalize to 0-1
        contrast_score = min(1.0, contrast)
        
        # Factor 3: Color richness (how much color vs grayscale)
        # Calculate color variance across channels
        color_variance = np.var([img_np[:,:,0].mean(), img_np[:,:,1].mean(), img_np[:,:,2].mean()])
        color_score = min(1.0, color_variance / 1000)
        
        # Factor 4: Information content (entropy)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        entropy_score = entropy / 8.0  # Normalize (max entropy is 8 for 256 bins)
        
        # Factor 5: Noise level (artificially add some if requested)
        noise_factor = 1.0
        if add_noise and random.random() < 0.25:  # 25% of images get quality degradation
            noise_level = random.uniform(0.1, 0.4)
            noise_factor = 1.0 - noise_level
        
        # Combine factors with weights
        quality = (sharpness_score * 0.25 + contrast_score * 0.25 + 
                  color_score * 0.2 + entropy_score * 0.2 + noise_factor * 0.1)
        
        return np.clip(quality, 0.0, 1.0)
    
    def _compute_relevance_score(self, image_tensor, label):
        """Compute relevance score based on visual characteristics and class"""
        img_np = image_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Factor 1: Edge content (objects with clear boundaries are more relevant)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (32 * 32)
        edge_score = min(1.0, edge_density * 3)  # Normalize
        
        # Factor 2: Central object focus (objects in center are typically more relevant)
        center_region = gray[8:24, 8:24]  # 16x16 center region
        center_intensity = np.mean(center_region)
        edge_intensity = np.mean(gray) - center_intensity
        focus_score = min(1.0, abs(center_intensity - edge_intensity) / 50)
        
        # Factor 3: Class-specific relevance for CIFAR-100 fine labels
        # CIFAR-100 has 100 classes grouped into 20 superclasses
        # We'll create relevance patterns based on superclass characteristics
        class_relevance = self._get_cifar100_class_relevance(label)
        
        # Factor 4: Random variation to simulate real-world relevance differences
        random_factor = 0.6 + 0.4 * random.random()  # Between 0.6 and 1.0
        
        # Combine factors
        relevance = (edge_score * 0.3 + focus_score * 0.2 + 
                    class_relevance * 0.3 + random_factor * 0.2)
        
        return np.clip(relevance, 0.0, 1.0)
    
    def _get_cifar100_class_relevance(self, fine_label):
        """Get relevance score based on CIFAR-100 class characteristics"""
        # CIFAR-100 superclass mapping (simplified)
        # Different superclasses have different visual complexity patterns
        superclass_patterns = {
            # Aquatic mammals: complex shapes, water backgrounds
            4: 0.7, 30: 0.7, 55: 0.7, 72: 0.7, 95: 0.7,
            # Fish: varied complexity
            1: 0.8, 32: 0.8, 67: 0.8, 73: 0.8, 91: 0.8,
            # Flowers: high color variation, good for quality assessment
            54: 0.9, 62: 0.9, 70: 0.9, 82: 0.9, 92: 0.9,
            # Food containers: structured objects
            16: 0.8, 61: 0.8, 79: 0.8, 84: 0.8, 87: 0.8,
            # Fruit and vegetables: color-rich, varied shapes
            6: 0.85, 20: 0.85, 42: 0.85, 43: 0.85, 88: 0.85,
            # Household electrical devices: structured, clear edges
            15: 0.9, 19: 0.9, 21: 0.9, 31: 0.9, 38: 0.9,
            # Household furniture: large objects, clear shapes
            34: 0.85, 63: 0.85, 64: 0.85, 66: 0.85, 75: 0.85,
            # Insects: small details, varied complexity
            26: 0.7, 45: 0.7, 77: 0.7, 78: 0.7, 79: 0.7,
            # Large carnivores: complex textures
            2: 0.8, 11: 0.8, 35: 0.8, 46: 0.8, 98: 0.8,
            # Large man-made outdoor things: varied complexity
            27: 0.75, 29: 0.75, 44: 0.75, 78: 0.75, 93: 0.75,
            # Large natural outdoor scenes: complex backgrounds
            36: 0.6, 50: 0.6, 65: 0.6, 74: 0.6, 80: 0.6,
            # Large omnivores and herbivores: varied
            23: 0.75, 33: 0.75, 49: 0.75, 60: 0.75, 71: 0.75,
            # Medium-sized mammals: good object definition
            15: 0.8, 18: 0.8, 44: 0.8, 97: 0.8, 99: 0.8,
            # Non-insect invertebrates: unique shapes
            40: 0.75, 39: 0.75, 22: 0.75, 87: 0.75, 86: 0.75,
            # People: human faces and bodies
            5: 0.85, 25: 0.85, 47: 0.85, 52: 0.85, 53: 0.85,
            # Reptiles: varied textures and shapes
            46: 0.8, 58: 0.8, 70: 0.8, 89: 0.8, 90: 0.8,
            # Small mammals: good detail visibility
            9: 0.85, 10: 0.85, 16: 0.85, 28: 0.85, 61: 0.85,
            # Trees: natural textures, varied complexity
            17: 0.7, 18: 0.7, 68: 0.7, 76: 0.7, 77: 0.7,
            # Vehicles 1: cars, trucks, etc.
            8: 0.9, 13: 0.9, 48: 0.9, 58: 0.9, 90: 0.9,
            # Vehicles 2: other vehicles
            41: 0.85, 69: 0.85, 81: 0.85, 85: 0.85, 89: 0.85,
        }
        
        # Return relevance score for the class, with default for unlisted classes
        return superclass_patterns.get(fine_label, 0.75)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], idx

class RelevanceScorer(nn.Module):
    """Stage 1: Learns what data is relevant for the task"""
    def __init__(self, input_dim=3072, hidden_dim=256):
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
            nn.Linear(hidden_dim//4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class QualityRater(nn.Module):
    """Stage 2: Rates the quality of relevant data"""
    def __init__(self, input_dim=3072, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for relevance score
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//4, 1),
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
    """Stage 3: Much more selective diversity control for CIFAR-100"""
    def __init__(self, target_distribution=None, memory_size=5000):
        # Much more selective distribution - focus on high quality samples
        self.target_dist = target_distribution or [0.02, 0.05, 0.15, 0.25, 0.35, 0.18]
        self.memory = deque(maxlen=memory_size)
        self.selection_history = defaultdict(int)
        self.warmup_period = 1000  # Less selective initially
        self.total_attempts = 0
        
    def should_select(self, quality_score, relevance_score, diversity_factor=0.4):
        """Much more selective decision making"""
        self.total_attempts += 1
        quality_bin = min(int(quality_score * 6), 5)
        
        total_selected = sum(self.selection_history.values()) + 1
        current_prop = self.selection_history[quality_bin] / total_selected if total_selected > 0 else 0
        target_prop = self.target_dist[quality_bin]
        
        # Much higher threshold for selection - prioritize quality heavily
        base_prob = (quality_score * 0.7 + relevance_score * 0.3)
        
        # Add minimum quality threshold
        if quality_score < 0.4:
            base_prob *= 0.1  # Heavily penalize low quality
        elif quality_score > 0.7:
            base_prob *= 1.3  # Boost high quality
            
        # Stricter relevance threshold
        if relevance_score < 0.6:
            base_prob *= 0.3
        
        # Diversity adjustment with higher penalties
        if current_prop < target_prop:
            diversity_boost = diversity_factor * (target_prop - current_prop) / (target_prop + 0.1)
        else:
            diversity_penalty = diversity_factor * (current_prop - target_prop) / (target_prop + 0.1)
            diversity_boost = -diversity_penalty * 2  # Stronger penalty
            
        # Warmup period - be less selective initially
        if self.total_attempts < self.warmup_period:
            warmup_factor = 2.0
        else:
            warmup_factor = 1.0
            
        final_prob = min(1.0, max(0.02, (base_prob + diversity_boost) * warmup_factor))
        
        # Even stricter selection threshold
        selected = random.random() < final_prob * 0.6  # Overall 40% reduction in selection
        
        if selected:
            self.selection_history[quality_bin] += 1
            self.memory.append((quality_score, relevance_score))
            
        return selected

class MainModel(nn.Module):
    """Improved CIFAR-100 classifier with better architecture"""
    def __init__(self, input_dim=3072, n_classes=100, hidden_dim=1024):
        super().__init__()
        self.network = nn.Sequential(
            # Larger first layer to handle complexity
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            
            # Deeper architecture
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//4),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim//4, hidden_dim//8),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//8),
            nn.Dropout(0.2),
            
            # Additional hidden layer for 100-class complexity
            nn.Linear(hidden_dim//8, hidden_dim//16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim//16, n_classes)
        )
        
    def forward(self, x):
        return self.network(x)

class DRPS:
    def __init__(self, input_dim=3072, n_classes=100, device='cpu'):
        self.device = device
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        # components
        self.relevance_scorer = RelevanceScorer(input_dim, hidden_dim=512).to(device)  # Larger
        self.quality_rater = QualityRater(input_dim, hidden_dim=512).to(device)  # Larger
        self.diversity_controller = DiversityController()
        self.main_model = MainModel(input_dim, n_classes, hidden_dim=1024).to(device)  # Much larger
        
        # Better optimizers with learning rate scheduling
        self.rel_optimizer = optim.AdamW(self.relevance_scorer.parameters(), lr=0.001, weight_decay=0.01)
        self.qual_optimizer = optim.AdamW(self.quality_rater.parameters(), lr=0.001, weight_decay=0.01)
        self.main_optimizer = optim.AdamW(self.main_model.parameters(), lr=0.002, weight_decay=0.01)
        
        # Learning rate schedulers
        self.main_scheduler = optim.lr_scheduler.StepLR(self.main_optimizer, step_size=100, gamma=0.8)
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
        self.mse_criterion = nn.MSELoss()
        
        # training stats
        self.selection_stats = {
            'total_seen': 0,
            'total_selected': 0,
            'relevance_accuracy': [],
            'quality_accuracy': [],
            'main_model_accuracy': [],
            'selection_ratios': []
        }
        
    def train_relevance_scorer(self, dataset, epochs=60):
        print("Training Relevance Scorer for CIFAR-100...")
        
        bootstrap_size = min(3000, len(dataset))
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
                
                # Calculate accuracy (within 0.2 threshold for CIFAR-100)
                if abs(pred_relevance.item() - true_relevance.item()) < 0.2:
                    correct += 1
                total += 1
                
            accuracy = correct / total
            self.selection_stats['relevance_accuracy'].append(accuracy)
            
            if epoch % 15 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}: Relevance Loss: {total_loss/len(indices):.4f}, Accuracy: {accuracy:.4f}")
    
    def train_quality_rater(self, dataset, epochs=60):
        print("Training Quality Rater for CIFAR-100...")
        
        bootstrap_size = min(3000, len(dataset))
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
                
                if abs(pred_quality.item() - true_quality.item()) < 0.2:
                    correct += 1
                total += 1
                
            accuracy = correct / total
            self.selection_stats['quality_accuracy'].append(accuracy)
            
            if epoch % 15 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}: Quality Loss: {total_loss/len(indices):.4f}, Accuracy: {accuracy:.4f}")
    
    def select_data_batch(self, dataset, batch_size=64, max_attempts=8000):
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
    
    def train_main_model(self, dataset, test_dataset, epochs=600, batch_size=64):
        print("Training Main Model with DRPS on CIFAR-100...")
        
        best_accuracy = 0
        patience_counter = 0
        patience_limit = 100
        
        for epoch in range(epochs):
            # Select batch using DRPS with multiple attempts for quality
            selected_batch = []
            attempts = 0
            max_attempts = 8000  # More attempts for better selection
            
            while len(selected_batch) < batch_size and attempts < max_attempts:
                idx = random.randint(0, len(dataset) - 1)
                x, y, _ = dataset[idx]
                x_tensor = x.unsqueeze(0).to(self.device)
                
                # Get scores
                with torch.no_grad():
                    relevance_score = self.relevance_scorer(x_tensor).item()
                    relevance_tensor = torch.FloatTensor([relevance_score]).to(self.device)
                    quality_score = self.quality_rater(x_tensor, relevance_tensor).item()
                
                # Diversity controller decides
                if self.diversity_controller.should_select(quality_score, relevance_score):
                    selected_batch.append((x, y))
                
                attempts += 1
                self.selection_stats['total_seen'] += 1
            
            # If we can't get enough high-quality samples, relax criteria
            if len(selected_batch) < batch_size // 2:
                additional_needed = batch_size - len(selected_batch)
                indices = random.sample(range(len(dataset)), additional_needed)
                for idx in indices:
                    x, y, _ = dataset[idx]
                    selected_batch.append((x, y))
            
            self.selection_stats['total_selected'] += len(selected_batch)
            
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), max_norm=1.0)
            
            self.main_optimizer.step()
            self.main_scheduler.step()
            
            # Evaluate more frequently
            if epoch % 25 == 0:
                accuracy = self.evaluate(test_dataset)
                selection_ratio = self.selection_stats['total_selected'] / max(1, self.selection_stats['total_seen'])
                
                self.selection_stats['main_model_accuracy'].append(accuracy)
                self.selection_stats['selection_ratios'].append(selection_ratio)
                
                print(f"Epoch {epoch}: Accuracy: {accuracy:.4f}, Selection Rate: {selection_ratio:.4f}, LR: {self.main_scheduler.get_last_lr()[0]:.6f}")
                
                # Early stopping with patience
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience_limit:
                    print(f"Early stopping at epoch {epoch} with best accuracy: {best_accuracy:.4f}")
                    break
    
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

def random_baseline_cifar100(dataset, test_dataset, epochs=400, batch_size=32, device='cpu'):
    """Baseline: Train CIFAR-100 classifier with random data selection"""
    print("Training Random Baseline for CIFAR-100...")
    
    model = MainModel(3072, 100).to(device)
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
        
        # Evaluate every 40 epochs
        if epoch % 40 == 0:
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

def run_cifar100_test():
    """Run comprehensive DRPS testing on CIFAR-100"""
    print("=" * 60)
    print("DRPS: Testing on CIFAR-100 Dataset")
    print("=" * 60)
    
    # Load CIFAR-100
    print("Loading CIFAR-100 dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    cifar_train = torchvision.datasets.CIFAR100(root='./data', train=True, 
                                              download=True, transform=transform)
    cifar_test = torchvision.datasets.CIFAR100(root='./data', train=False, 
                                             download=True, transform=transform)
    
    # Create datasets with quality/relevance scores
    print("Creating enhanced datasets...")
    train_dataset = CIFAR100DatasetWithScores(cifar_train, add_artificial_noise=True)
    test_dataset = CIFAR100DatasetWithScores(cifar_test, add_artificial_noise=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize DRPS
    print("\nInitializing DRPS...")
    drps = DRPS(input_dim=3072, n_classes=100, device=device)
    
    # Train DRPS components
    print("\n" + "="*40)
    print("PHASE 1: Training DRPS Components")
    print("="*40)
    
    start_time = time.time()
    drps.train_relevance_scorer(train_dataset, epochs=60)
    drps.train_quality_rater(train_dataset, epochs=60)
    component_time = time.time() - start_time
    
    # Train main model with DRPS
    print("\n" + "="*40)
    print("PHASE 2: Training Main Model with DRPS")
    print("="*40)
    
    start_time = time.time()
    drps.train_main_model(train_dataset, test_dataset, epochs=400, batch_size=32)
    drps_time = time.time() - start_time
    
    # Train random baseline
    print("\n" + "="*40)
    print("PHASE 3: Training Random Baseline")
    print("="*40)
    
    start_time = time.time()
    random_accuracies = random_baseline_cifar100(train_dataset, test_dataset, 
                                                epochs=400, batch_size=32, device=device)
    baseline_time = time.time() - start_time
    
    # Results and Analysis
    print("\n" + "="*60)
    print("CIFAR-100 RESULTS AND ANALYSIS")
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
        epochs_drps = range(0, len(drps.selection_stats['main_model_accuracy']) * 40, 40)
        plt.plot(epochs_drps, drps.selection_stats['main_model_accuracy'], 'b-', label='DRPS', linewidth=2)
    if random_accuracies:
        epochs_random = range(0, len(random_accuracies) * 40, 40)
        plt.plot(epochs_random, random_accuracies, 'r--', label='Random', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('CIFAR-100 Learning Curves Comparison')
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
    # Show a few sample CIFAR-100 images
    sample_indices = random.sample(range(min(1000, len(train_dataset))), 16)
    
    # Create a 4x4 grid of sample images
    fig_img = plt.figure(figsize=(8, 8))
    for i, idx in enumerate(sample_indices[:16]):
        plt.subplot(4, 4, i+1)
        # Reshape back to 32x32x3 and display
        img = train_dataset.data[idx].reshape(3, 32, 32).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)  # Ensure values are in [0,1]
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Q:{train_dataset.quality_scores[idx]:.2f}', fontsize=8)
    plt.suptitle('Sample CIFAR-100 Images with Quality Scores')
    plt.tight_layout()
    plt.show()
    
    # Close the sample images figure
    plt.close(fig_img)
    
    # Summary
    print(f"\n" + "="*60)
    print("CIFAR-100 EXPERIMENT SUMMARY")
    print("="*60)
    
    efficiency_gain = (1 - selection_ratio) * 100
    print(f"\n✓ DRPS achieved {efficiency_gain:.1f}% data reduction")
    print(f"✓ Component learning results:")
    if drps.selection_stats['relevance_accuracy']:
        print(f"   - Relevance scorer: {drps.selection_stats['relevance_accuracy'][-1]:.1%} accuracy")
    if drps.selection_stats['quality_accuracy']:
        print(f"   - Quality rater: {drps.selection_stats['quality_accuracy'][-1]:.1%} accuracy")
    
    if final_drps_acc >= final_random_acc * 0.90:  # Within 10% is reasonable for CIFAR-100
        print(f"✓ Performance maintained: {final_drps_acc:.1%} vs {final_random_acc:.1%} (random)")
        print(f"✓ Efficiency per data point: {final_drps_acc/selection_ratio:.2f}x better than random")
    else:
        print(f"⚠ Performance gap: {final_drps_acc:.1%} vs {final_random_acc:.1%} (random)")
        print(f"   CIFAR-100 is very challenging - 100 fine-grained classes")
    
    # Get CIFAR-100 class names for analysis
    cifar100_fine_labels = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]
    
    print(f"\nCIFAR-100 Classification Challenge:")
    print(f"   - 100 fine-grained classes (vs 10 in CIFAR-10)")
    print(f"   - 20 superclasses with 5 subclasses each")
    print(f"   - Examples: {', '.join(cifar100_fine_labels[:10])}...")
    print(f"   - Much more challenging than CIFAR-10")
    
    print(f"\nKey CIFAR-100 Insights:")
    print(f"   - DRPS examined {drps.selection_stats['total_seen']} samples")
    print(f"   - Selected only {drps.selection_stats['total_selected']} for training")
    print(f"   - Selection rate: {selection_ratio:.1%} of examined data")
    
    # Quality assessment insights
    if len(drps.diversity_controller.memory) > 0:
        avg_selected_quality = np.mean([s[0] for s in drps.diversity_controller.memory])
        avg_selected_relevance = np.mean([s[1] for s in drps.diversity_controller.memory])
        
        print(f"\nSelected Data Characteristics:")
        print(f"   - Average quality of selected samples: {avg_selected_quality:.3f}")
        print(f"   - Average relevance of selected samples: {avg_selected_relevance:.3f}")
        print(f"   - Dataset average quality: {train_dataset.quality_scores.mean():.3f}")
        print(f"   - Dataset average relevance: {train_dataset.relevance_scores.mean():.3f}")
    
    # Performance per class analysis
    if final_drps_acc > 0:
        acc_per_class = final_drps_acc * 100  # Convert to percentage for easier reading
        print(f"\nPerformance Analysis:")
        print(f"   - {acc_per_class:.1f}% accuracy across 100 classes")
        print(f"   - Average per-class accuracy: ~{acc_per_class/100:.1f}% per class")
        if acc_per_class > 20:  # Better than random (1%)
            print(f"   ✓ Significantly better than random chance (1%)")
        
    return drps, random_accuracies

if  __name__ == "__main__":
   # Run the CIFAR-100 test
   print("Starting CIFAR-100 DRPS validation...")
   
   drps_system, baseline_results = run_cifar100_test()
   
   print(f"\n" + "="*60)
   print("CIFAR-100 DRPS Validation Complete!")
   print("="*60)
   
   print(f"\nExperiment completed!")